import argparse
import os
import sys

import jax
from jax._src.config import config
import jax.numpy as np
import jax_cosmo as jc
from astropy.io import fits
from jax_cosmo.power import linear_matter_power, nonlinear_matter_power
from jax_cosmo.scipy.interpolate import interp
from scipy.special import spherical_jn

config.update("jax_enable_x64", True)

# This is the last compiled version of class, not necessarily the one used in MCMC
# from classy import Class


# Here we list all the constants used in the likelihood analysis
# It is the only place where they are written
constants = {
    "JoulestoErg": 1.0e7,
    "metersTocm": 1.0e2,
    "metersToMpc": 3.2408e-23,
    "gravityConstantG": 6.673e-11,  # G in m^3/ s^2 / kg
    "speedOfLightC": 299792458.0,  # c in m / s
    "cMpcInvSec": 9.72e-15,  # Speed of light in Mpc/sec
}


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def create_array(start, stop, num, spacing):
    """Creates an array with linear or log spacing

    Args:
        start (float): Minimum value of array
        stop (float): Maximum value of array
        num (int): Number of points in array
        spacing ({'linear', 'log'}): the type of spacing between samples

    Raises:
        ValueError: If the spacing is not recognized.

    Returns:
        ndarray: an array of values consistent with input arguments
    """
    if spacing == "linear":
        array = np.linspace(start, stop, num)
    elif spacing == "log":
        array = np.logspace(np.log10(start), np.log10(stop), num)
    else:
        raise ValueError("Spacing not recognized!")
    return array


def nu_l(ell):
    """The position x=nu of the maximum value of the spherical Bessel of order ell.

    Args:
        ell (int): The order of the spherical Bessel

    Returns:
        int: The approximate argmax of the spherical Bessel of order ell.
    """
    return ell + 1.0 / 2.0


def get_x_full(ell_max, x_min, after, points_pp):
    r"""Get all values of the product $$x=k*chi$$ in the non-rectangular k-chi grid

    Args:
        ell_max (int): The maximum order ell of the spherical Bessel.
        x_min (float): The minimum value of the x_full array.
        after (int): Number of periods after the first maximum of the $$j_{\ell_max}$$.
                     Used to generate x_max.
        points_pp (int): Points per period of the spherical Bessel.
                         Used to generate the linear spacing.

    Returns:
        ndarray: The full array of x values. The spherical Bessel of each order
                 will be evaluated on parts of this array.
    """
    x_max = nu_l(ell_max) + after * 2.0 * np.pi
    x_num = (x_max - x_min) / (2 * np.pi) * points_pp
    # convert to int to avoid floating point errors
    x_num = np.round(x_num).astype(np.int32)
    x_full = create_array(x_min, x_max, x_num, "linear")
    return x_full


def xlim_l(ell, x_full, before, after):
    r"""Return the indices for `x_min` and `x_max` in `x_full` for a value of ell,
    given a number of requested periods before and after.

    Args:
        ell (int): The order of the spherical Bessel
        x_full (ndarray): One dimensional array of values of $$x=k\chi$$. This
                          corresponds to the full non-rectangular grid.
        before (int): Number of periods before the first max of the spherical Bessel.
        after (int): Number of periods after the first max of the spherical Bessel.

    Returns:
        tuple: A tuple of indices (x_min_idx, x_max_idx) for the spherical Bessel of order ell
    """
    x_min = nu_l(ell) - before * 2.0 * np.pi
    x_max = nu_l(ell) + after * 2.0 * np.pi
    x_min_idx = np.searchsorted(x_full, x_min, side="left")
    x_max_idx = np.searchsorted(x_full, x_max, side="left")
    return x_min_idx, x_max_idx


def get_intermediate_grids(k_vec, x_full, k_sparse):
    r"""Generate grids that do not require the cosmology.

    Args:
        k_vec (ndarray): 1D wavenumber array for  the non-rectangular grid
        x_full (ndarray): The full 1D array of x values where $$x=k*\chi$$.
        k_sparse (ndarray): 1D wavenumber array for the sparse grid.

    Returns:
        dict: A dictionary containing intermidate grids:
               - chi_grid: 2D grid for the comoving distance,
                           defined such that `chi_grid[i,j] = x[i]/k[j]`
               - k_mid: 1D array at the midpoints of k_sparse.
                        Generated to evaluate there quantities of the
                        sparse grid.
               - k_indices_grid: indices that map points of the rectangular
                                 grid to points of the non-rectangular grid.
    """
    intermediate = {}
    intermediate["chi_grid"] = x_full[None, :] / k_vec[:, None]
    intermediate["k_mid"] = (k_sparse[1:] + k_sparse[:-1]) / 2.0

    k_indices = np.digitize(k_vec, k_sparse)
    intermediate["k_indices_grid"] = np.tile(k_indices[:, None], (1, len(x_full)))
    return intermediate


def biasFactor(bNow, z_vec):
    r"""Compute the bias using the expression $$b_{Now}*\sqrt{1+z}$$

    Args:
        bNow (float): The bias at z=0
        z_vec (float or ndarray): Redshift value(s) where the bias is evalueated

    Returns:
        float or ndarray: the array of bias values
    """
    return bNow * np.sqrt(1 + z_vec)


def rho_crit(cosmo):
    """Calculates the critical energy density at present time
    density in erg/cm^3 for a given cosmology
    rho_crit = 3*H0**2 * c^2 /(8 pi G)

    Args:
        cosmo (Class): An instance of the computed cosmology

    Returns:
        float: Critical density at present time in erg/cm^3
    """
    HubbleInvMpcNow = cosmo.h * 100 * constants["metersToMpc"] / constants["cMpcInvSec"]
    conversions = (
        constants["metersToMpc"] ** 2 / constants["metersTocm"] ** 3
    ) * constants["JoulestoErg"]
    rhoCritical = (
        conversions
        * 3.0
        * HubbleInvMpcNow**2
        * constants["speedOfLightC"] ** 4
        / (8.0 * np.pi * constants["gravityConstantG"])
    )
    return rhoCritical


# def tau_from_z(cosmology, z_vec):
#     """
#     NOTE: commented out because \chi is used instead of \tau
#     get conformal time from redshift
#     """
#     zPoints = cosmology.get_background()['z']
#     tauPoints = cosmology.get_background()['conf. time [Mpc]']
#     tauVec = interp1d(zPoints, tauPoints, kind='cubic')(z_vec)
#     return tauVec


# def chi_from_z(cosmo, z_vec):
#     """Converts redshift z to comoving distance \chi

#     Args:
#         cosmo (Class): An instance of the computed cosmology
#         z_vec (ndarray): Redshift value(s) to be converted

#     Returns:
#         ndarray: Value(s) of comoving distance
#     """
#     z_points = cosmo.get_background()['z']
#     chi_points = cosmo.get_background()['comov. dist.']
#     chi_vec = interp1d(z_points, chi_points, kind='cubic')(z_vec)
#     return chi_vec


def chi_from_z(cosmo, z_vec):
    """Calculates the comoving distance in Mpc for a given redshift

    Args:
        cosmo (Class): An instance of the computed cosmology

    Returns:
        float: Comoving distance in Mpc
    """
    chi = jc.background.radial_comoving_distance(cosmo, 1.0 / (1.0 + z_vec))
    return chi


def z_from_chi(cosmo, chi_vec):
    """Calculates the comoving distance in Mpc for a given redshift

    Args:
        cosmo (Class): An instance of the computed cosmology

    Returns:
        float: Comoving distance in Mpc
    """
    # create an array of z values to interpolate
    z = np.linspace(0, 10, 1000)
    chi = chi_from_z(cosmo, z)
    z_vec = interp(chi_vec, chi, z)
    return z_vec


# def chi_from_tau(cosmology, tau_vec):
#     """
#     get conformal distance chi from conformal time tau
#     """
#     return cosmology.get_background()['conf. time [Mpc]'][-1] - tau_vec


# def z_from_chi(cosmo, chi_vec):
#     """Converts comoving distance \chi to redshift z

#     Args:
#         cosmo (Class): An instance of the computed cosmology
#         chi_vec (ndarray): Value(s) of comoving distance to be converted

#     Returns:
#         ndarray: Value(s) of redshift
#     """
#     chi_points = cosmo.get_background()['comov. dist.']
#     z_points = cosmo.get_background()['z']
#     z_vec = interp1d(chi_points, z_points, kind='cubic')(chi_vec)
#     return z_vec


# def HubbleZ(cosmo, z_vec):
#     # One could use the cosmo.get_background()["H [1/Mpc]"] array, and create and interpolator,
#     # but cosmo.Hubble(z) probably does exactly that.
#     ## Gives Hubble in units of 1/Mpc.
#     ## To get the usual value of H in km/sec/Mpc multiply by c in km/sec
#     return np.array(map(cosmo.Hubble,z_vec))


# def vec_sph_bessel():
#     """
#       MEMORY ERROR
#     vectorized spherical Bessel with respect to the argument l
#     """
#     vSphBessel = np.vectorize(spherical_jn,excluded={1,})
#     return vSphBessel # the argument z has nothing to do with redshift


def Kernel_A_of_z_and_f_gauss(z_vec, f, A1, z_mean, z_sigma):
    r"""Returns a parametrized astrophysical kernel $$\mathcal{A}$$ as a
    function of redshift. It does not depend on the frequency

    Args:
        z_vec (ndarray): Array of redshifts where the kernel is evaluated
        f (float): the value of frequency (currently unused)
        A1 (float): The amplitude of the Gaussian
        z_mean (float): The position of the maximum of the Gaussian
        z_sigma (float): The standard deviation of the Gaussian

    Returns:
        ndarray: The parametrized Gaussian astrophysical kernel
    """
    # TODO: the frequency fully specifies the other parameters, so
    # we should create interpolators for A_1, z_mean and z_sigma
    # and remove them from being parameters of this function
    # OR (better) remove f from parameter, but evaluate the others outside of this
    # function as a function of frequency
    kernel = A1 * np.exp(-((z_vec - z_mean) ** 2) / (2 * z_sigma**2))
    return kernel


def get_deltaM_array(cosmo, k_vec, z_vec, nonlinear="Halofit"):
    """Returns a 2D array containing the matter
    overdensities as a function of k and z.s
    First index corresponds to k, second index to z
    This is used both to generate the data and to
    run the likelihood.

    Args:
        cosmo (Class): An instance of the computed cosmology
        k_vec (ndarray): Wavenumber vector at which delta_M is computed
        z_vec (ndarray): Redshift vector at which delta_M is computed
        nonlinear (str, optional): Request non-linear Pk if "Halofit".
                                   Defaults to "Halofit".

    Returns:
        ndarray: 2D array of delta_M values. Size (len(k_vec), len(z_vec))
    """
    isnonlinear = True if nonlinear == "Halofit" else False

    # matterPk = cosmo.get_pk_array(k_vec, z_vec,
    #                               len(k_vec), len(z_vec),
    #                               nonlinear=isnonlinear)

    # when reshaping, "F" means to read / write the elements using Fortran-like
    # index order, with the first index changing fastest, and the last index
    # changing slowest. CLASS returns Pk as a 1D array with k (the first index)
    # running faster.
    # matterPk = np.reshape(matterPk, (len(k_vec),-1), 'F')
    k_vec = k_vec.reshape(-1, 1)
    z_vec = z_vec.T
    if isnonlinear:
        matterPk = nonlinear_matter_power(cosmo, k_vec, z_vec)
    else:
        matterPk = linear_matter_power(cosmo, k_vec, z_vec)

    deltaM = np.sqrt(matterPk)
    return deltaM


def get_cosmo_eff(cosmo, z_sparse, intermediate_grids, args, nonlinear):
    r"""Computes bias and $$\delta_M$$ on the rectangular grid and interpolate
    its values on the non-rectangular grid. Additionally computes grids
    needed for the calculation of the astrophysical kernel.

    Args:
        cosmo (Class): Instance of the computed cosmology
        z_sparse (ndarray): Array of redshifts in the sparse grid
        intermediate_grids (dict): Dictionary of grids calculated before
                                   the cosmology is initialized. They are
                                   generated by the `get_intermediate_grids()`
                                   function.
        args (Argparser): Arguments Parser Instance. Specifically,
                          z_min, z_max, b0 come from these args.
        nonlinear (str): Request non-linear Pk if "Halofit", linear otherwise.

    Returns:
        tuple: bias, deltaM, assorted_grids.
               The returned grids are needed for the calculation of the
               astrophysical kernel. Its computation is separate, because it
               depends on the GW frequency.

    """

    # Get intermediate grid
    chi_grid = intermediate_grids["chi_grid"]
    k_mid = intermediate_grids["k_mid"]
    k_indices_grid = intermediate_grids["k_indices_grid"]

    # Get sparse chi/z grids
    chi_sparse = chi_from_z(cosmo, z_sparse)
    chi_mid = (chi_sparse[1:] + chi_sparse[:-1]) / 2.0
    z_mid = z_from_chi(cosmo, chi_mid)

    chi_min, chi_max = [chi_from_z(cosmo, z) for z in (args.z_min, args.z_max)]
    chi_mask = np.logical_or(chi_grid < chi_min, chi_grid > chi_max)
    chi_indices = np.digitize(chi_grid, chi_sparse)
    chi_valid_ind = chi_indices[~chi_mask] - 1
    k_valid_ind = k_indices_grid[~chi_mask] - 1

    b_z = biasFactor(args.b0, z_mid)
    deltaM_k_z = get_deltaM_array(cosmo, k_vec=k_mid, z_vec=z_mid, nonlinear=nonlinear)

    b_eff = np.full_like(chi_indices, dtype=float, fill_value=np.nan)
    deltaM_eff = np.full_like(chi_indices, dtype=float, fill_value=np.nan)

    # jax arrays are immutable, so we need to create a new array
    # b_eff[~chi_mask] = b_z[chi_valid_ind]
    b_eff = b_eff.at[~chi_mask].set(b_z[chi_valid_ind])

    # deltaM_eff[~chi_mask] = deltaM_k_z[k_valid_ind, chi_valid_ind]
    deltaM_eff = deltaM_eff.at[~chi_mask].set(deltaM_k_z[k_valid_ind, chi_valid_ind])

    # grids needed to calculate A_eff. The calculation is done
    # separately, because it is frequency dependent
    assorted_grids = {
        "z_mid": z_mid,
        "chi_indices": chi_indices,
        "chi_mask": chi_mask,
        "chi_valid_ind": chi_valid_ind,
    }

    return b_eff, deltaM_eff, assorted_grids


def compute_kernel_on_grid(
    params,
    freq,
    assorted_grids,
    args,
    not_chi_mask_nonzero=None,
    A_kernel_interp2d=None,
):
    """Wrapper for the computation of the astrophysical kernel.
    Frequency has to be a number.

    Args:
        freq (int): The frequency at which the kernel is calculated
        assorted_grids (dict): Grids that are needed for the calculation.
                               They are the output of compute_cosmo_eff
        args (Argparser): Arguments Parser Instance. Specifically,
                          `A_max`, `mean_z`, `sigma_z`
                          and `overwriteKernel` come from these args.
        A_kernel_interp2d (ndarray, optional): The values of the astrophysical kernel.
                                               Used only if args.overwriteKernel==False
                                               Defaults to None.

    Raises:
        ValueError: If we are not overwriting kernel AND we are not supplying
                    the kernel values as an ndarray.

    Returns:
        ndarray: The values of the astrophysical kernel on the non-rectangular grid
    """

    z_mid = assorted_grids["z_mid"]
    chi_indices = assorted_grids["chi_indices"]
    chi_mask = assorted_grids["chi_mask"]
    chi_valid_ind = assorted_grids["chi_valid_ind"]

    if args.overwriteKernel:
        A_z = Kernel_A_of_z_and_f_gauss(z_mid, freq, *params)
    else:
        if A_kernel_interp2d is None:
            raise ValueError("Kernel from data was not found")
        else:
            A_z = A_kernel_interp2d(z_mid, freq)

    A_eff = np.full_like(chi_indices, dtype=float, fill_value=np.nan)

    A_at_valid_index = A_z.at[chi_valid_ind].get()

    # not_chi_mask = ~chi_mask
    # print(A_at_valid_index.shape)
    # print(not_chi_mask.nonzero()[0].shape, not_chi_mask.nonzero()[1].shape)
    # print(not_chi_mask.shape)
    # print(not_chi_mask.sum())
    # selected_A_eff = A_eff.at[not_chi_mask]
    # A_eff = selected_A_eff.set(A_at_valid_index)
    # print(A_eff.shape)
    if not_chi_mask_nonzero is None:
        not_chi_mask = ~chi_mask
        not_chi_mask_nonzero = not_chi_mask.nonzero()
        A_eff = A_eff.at[not_chi_mask_nonzero].set(A_at_valid_index)
    A_eff = A_eff.at[not_chi_mask_nonzero].set(A_at_valid_index)
    # A_eff = np.where(not_chi_mask, A_at_valid_index, A_eff)

    return A_eff, A_z


# def make_sparse(l_vec, intervals=[30,40,240,1000], sample_distances=[1,10,20,40]):
#     """If intervals is not sorted it will be sorted.
#     TODO: The default values probably need refinement.

#     Args:
#         l_vec (ndarray): The array of all ell at which the likelihood is evaluated
#         intervals (list, optional): The upper limit of each interval.
#                                     Each interval has different sampling rate.
#                                     Defaults to [30,40,240,1000].
#         sample_distances (list, optional): The distance between samples in each interval.
#                                            Defaults to [1,10,20,40].

#     Raises:
#         ValueError: If the specifications of the interval are inconsistent.

#     Returns:
#         ndarray: array of sparse ell.
#     """
#     if (not isinstance(intervals, list))  or (not isinstance(sample_distances, list)):
#         raise ValueError("intervals and samples_distances should both be lists.")
#     if len(intervals) != len(sample_distances):
#         raise ValueError("Arrays `intervals` and `sample_distances` should have same length")
#     if any(x < 1 for x in sample_distances):
#         raise ValueError("All sample distances have to be greater than 1.")

#     l_min, l_max = np.min(l_vec), np.max(l_vec)

#     # if np.max(np.array(intervals)).astype(int) < l_max.astype(int):
#     #     raise ValueError("Not sure how to deal with the interval from {} to {}".format(np.max(intervals),l_max))

#     intervals = sorted(intervals)

#     ell_value = l_min
#     ell_list = []
#     interv_idx = 0
#     while interv_idx < len(intervals):
#         while ell_value < intervals[interv_idx] and ell_value <= l_max:
#             ell_list.append(ell_value)
#             ell_value += sample_distances[interv_idx]
#         interv_idx += 1
#     if l_max not in ell_list:
#         ell_list.append(l_max)

#     l_sparse_vec = np.array(ell_list)

#     return l_sparse_vec


def make_sparse(
    l_max, l_min=0, intervals=[30, 40, 240, 1000], sample_distances=[1, 10, 20, 40]
):
    """If intervals is not sorted it will be sorted.
    TODO: The default values probably need refinement.

    Args:
        l_vec (ndarray): The array of all ell at which the likelihood is evaluated
        intervals (list, optional): The upper limit of each interval.
                                    Each interval has different sampling rate.
                                    Defaults to [30,40,240,1000].
        sample_distances (list, optional): The distance between samples in each interval.
                                           Defaults to [1,10,20,40].

    Raises:
        ValueError: If the specifications of the interval are inconsistent.

    Returns:
        ndarray: array of sparse ell.
    """
    if (not isinstance(intervals, list)) or (not isinstance(sample_distances, list)):
        raise ValueError("intervals and samples_distances should both be lists.")
    if len(intervals) != len(sample_distances):
        raise ValueError(
            "Arrays `intervals` and `sample_distances` should have same length"
        )
    if any(x < 1 for x in sample_distances):
        raise ValueError("All sample distances have to be greater than 1.")

    # if np.max(np.array(intervals)).astype(int) < l_max.astype(int):
    #     raise ValueError("Not sure how to deal with the interval from {} to {}".format(np.max(intervals),l_max))

    intervals = sorted(intervals)

    ell_value = l_min
    ell_list = []
    interv_idx = 0
    while interv_idx < len(intervals):
        upper_bound = intervals[interv_idx]
        while ell_value < upper_bound and ell_value <= l_max:
            ell_list.append(ell_value)
            ell_value += sample_distances[interv_idx]
        interv_idx += 1
    if l_max not in ell_list:
        ell_list.append(l_max)

    l_sparse_vec = np.array(ell_list)

    return l_sparse_vec


def write_sph_bessel(fname, l_vec, x_vec, before, after):
    r"""Writes the values of the spherical and the grid specifications to a file.

    Args:
        fname (str): Relative or absolute path to the file.
        l_vec (ndarray): The 1D array of $$\ell$$ values where $$j_\ell$$ is stored.
                         Can be sparse.
        x_vec (ndarray): 1D array of x values where the computations $$j_\ell(x)$$
                         will take place. Each $$\ell$$ uses different part of this grid.
        before (int): The number of periods before the first maximum of each $$j_\ell$$
        after (int or callable): The number of periods after the first maximum
                                 of each $$j_\ell$$. Can be a function of $$\ell$$.


    Raises:
        IndexError: If x_vec is too small given the number of periods
                    requested for \ell_{max}.

    Returns:
        warning: Currently always False, but can be modified.
    """
    warning = False

    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.writeto(fname, overwrite=True)

    write_to_fits(fname, l_vec, "l", type="image", header=None)

    l_max = l_vec.max()
    after_max = after(l_max) if callable(after) else after
    x_max_idx_lmax = xlim_l(l_max, x_vec, before=before, after=after_max)[1]
    if x_max_idx_lmax == len(x_vec):
        raise IndexError(
            "x vector is too small compared to the requested number of periods after l_max. Increase x_max."
        )

    write_to_fits(fname, x_vec, "x", type="image", header=None)
    len_x = len(x_vec)
    write_to_fits(
        fname, np.array([len_x], dtype=int), "len_x", type="image", header=None
    )

    # Append spherical Bessel functions to HDUList and write the to FITS
    with fits.open(fname, mode="update") as hdul:
        for ell in l_vec:
            print("ell", ell, end="\r")
            if callable(after):
                after_n = after(ell)
            else:
                after_n = after
            x_min_idx, x_max_idx = xlim_l(ell, x_vec, before=before, after=after_n)
            # Write x, bessel_j vectors in the FITS file
            x_name = "x_{}_idx".format(ell)
            name = "bessel_{}".format(ell)
            bessel_x = spherical_jn(n=ell, z=x_vec[x_min_idx : x_max_idx + 1])
            x_image = fits.ImageHDU(
                np.array([x_min_idx, x_max_idx], dtype=int), name=x_name, header=None
            )
            bessel_image = fits.ImageHDU(bessel_x, name=name, header=None)
            try:
                hdul[x_name] = x_image
                hdul[name] = bessel_image
            except KeyError:
                hdul.append(x_image)
                hdul.append(bessel_image)
    print("\n")

    return warning


def get_bessel_x_l(fname, l_vec):
    r"""Retrieve values of the spherical Bessel from file.
    The grid specifications restrict the points x where $$j_\ell(x)$$ is non-zero.

    Args:
        fname (str): Relative or absolute path to the file.
        l_vec (ndarray): The 1D array of $$\ell$$ values where $$j_\ell$$ is computed.
                         Should be sparse to speed up calculation.

    Raises:
        NotImplementedError: Raises error when $$x=0$$ is included in the data.
                             This is not an issue, since $$\chi$$ has a non-zero limit
                             and $$k$$ is log-sampled.

    Returns:
        ndarray: The 2D array (x,ell) of the spherical Bessel.
    """
    hdul = fits.open(fname)

    len_x = int(hdul["len_x"].data)
    bessel_x_l = np.zeros((len_x, len(l_vec)))
    for lindex, ell in enumerate(l_vec):
        print("ell", ell, end="\r")
        # x = hdul['x_{}'.format(l)].data
        x_min_idx, x_max_idx = hdul["x_{}_idx".format(ell)].data
        # bessel_x_l[x_min_idx:x_max_idx+1, lindex] = hdul['bessel_{}'.format(l)].data
        bessel_x_l = bessel_x_l.at[x_min_idx : x_max_idx + 1, lindex].set(
            hdul["bessel_{}".format(ell)].data
        )
        if hdul["x"].data.min() == 0:
            # none should be zero
            raise NotImplementedError
    print("\n")

    return bessel_x_l


def simps_coeffs(h, h_idxp1):
    """Computes the coefficients needed for irregular Simpson's rule.

    Args:
        h (ndarray): step sizes, possibly irregular.
        h_idxp1 (ndarray): step sizes transposed by one.

    Returns:
        tuple: alpha, beta and gamma ndarray objects
    """
    alpha_numer = 2 * h_idxp1**3 - h**3 + 3 * h * h_idxp1**2
    alpha_denom = 6 * h_idxp1 * (h_idxp1 + h)

    beta_numer = h_idxp1**3 + h**3 + 3 * h_idxp1 * h * (h_idxp1 + h)
    beta_denom = 6 * h_idxp1 * h

    eta_numer = 2 * h**3 - h_idxp1**3 + 3 * h**2 * h_idxp1
    eta_denom = 6 * h * (h_idxp1 + h)

    alpha = alpha_numer / alpha_denom
    beta = beta_numer / beta_denom
    eta = eta_numer / eta_denom

    return alpha, beta, eta


def integrate_vec(integrand, steps):
    r"""Works for (k,x,ell) over chi and (k,ell) over k integrands
    Examples of steps:
    k_steps = np.diff(k_vec)
    chi_step = np.diff(chi_grid_unfiltered, axis=1)

    """
    dims = len(integrand.shape)
    steps_idxp0 = steps[..., 0::2]
    steps_idxp1 = steps[..., 1::2]
    # if odd number of subintervals, remove last interval
    # better would be to handle differently the last few intervals.
    if steps_idxp0.size > steps_idxp1.size:
        steps_idxp0 = steps_idxp0[..., :-1]
    alpha, beta, eta = simps_coeffs(steps_idxp0, steps_idxp1)
    integrand_idxp0 = integrand[..., 0:-1:2, :]
    integrand_idxp1 = integrand[..., 1::2, :]
    integrand_idxp2 = integrand[..., 2::2, :]
    if integrand_idxp0.size > integrand_idxp2.size:
        integrand_idxp0 = integrand_idxp0[..., :-1, :]
    if integrand_idxp1.size > integrand_idxp2.size:
        integrand_idxp1 = integrand_idxp1[..., :-1, :]
    axis = dims - 2
    integral = np.sum(
        alpha[..., None] * integrand_idxp2
        + beta[..., None] * integrand_idxp1
        + eta[..., None] * integrand_idxp0,
        axis=axis,
    )
    return integral


def compute_clustering_cl(cosmo, A_eff, b_eff, deltaM_eff, bessel_x_l, chi_grid, k_vec):
    r"""[summary]

    Args:
        cosmo (Class): [description]
        f (float): [description]
        A_eff (ndarray): [description]
        b_eff (ndarray): [description]
        deltaM_eff (ndarray): [description]
        bessel_x_l (ndarray): [description]
        chi_grid (ndarray): [description]
        k_vec (ndarray): [description]

    Returns:
        ndarray: 1D array containing clustering $$C_\ell$$
    """
    integrand_chi = (
        1
        / (4.0 * np.pi * rho_crit(cosmo) * constants["cMpcInvSec"])
        * (A_eff * b_eff * deltaM_eff)[..., None]
        * bessel_x_l
    )
    # integrand_chi[np.isnan(integrand_chi)]=0
    # integrand_chi = integrand_chi.at[np.isnan(integrand_chi)].set(0)
    integrand_chi = np.where(np.isnan(integrand_chi), 0, integrand_chi)
    chi_step = np.diff(chi_grid, axis=1)
    integral_chi = integrate_vec(integrand_chi, chi_step)

    k_step = np.diff(k_vec)
    integrand_k = 2 / np.pi * k_vec[:, None] ** 2 * integral_chi**2
    clustering = integrate_vec(integrand_k, k_step)

    return clustering


def compute_spatial_shot_noise(cosmo, A_z, chi_vec, n_G):
    rho_c = rho_crit(cosmo)
    c_Mpc = constants["cMpcInvSec"]
    # TODO: chack if trapezoidal rule is good enough
    integral = np.trapz(
        y=(A_z / rho_c / c_Mpc) ** 2 * 1 / chi_vec**2 * 1 / n_G, x=chi_vec
    )
    spatial_noise = (1 / 4 * np.pi) ** 2 * integral
    return spatial_noise


def interpolate_cl(cl, l_sparse, l_vec):
    r"""[summary]

    Args:
        cl ([type]): [description]
        l_sparse ([type]): [description]
        l_vec ([type]): [description]

    Returns:
        ndarray: 1D array contatining $$C_\ell$$, interolated over $$\ell$$
    """
    # z = jnp.linspace(0, 10, 1000)
    # chi = chi_of_z(cosmo, z)
    # z_vec = interp(chi_vec, chi, z)
    cl_vec = interp(l_vec, l_sparse, cl)
    return cl_vec


def compute_loglkl_old(data_cl, theory_cl, l_vec):
    # Combining theory_cl and data_cl to calculate the likelihood
    # Using equation (3) of arXiv 1811.11584
    # Note that most of the expression below can be precomputed if needed
    # Note that data_cl already have noise inside

    chi2_l = (
        2.0
        * (2.0 * l_vec + 1.0)
        / 2.0
        * ((data_cl / theory_cl) + np.log(2.0 * np.pi * theory_cl))
    )
    # Exclude l = 0
    if l_vec[0] == 0:
        chi2_l = chi2_l[1:]
    chi2 = np.sum(chi2_l)
    loglklhood = -0.5 * chi2
    return loglklhood


def compute_loglkl(data_cl, theory_cl, l_vec):
    # Combining theory_cl and data_cl to calculate the likelihood
    # Using equation (3) of arXiv 1811.11584
    # Note that most of the expression below can be precomputed if needed
    # Note that data_cl already have noise inside

    chi2_l = (2.0 * l_vec + 1.0) * ((data_cl / theory_cl) + np.log(theory_cl)) - (
        2.0 * l_vec - 1.0
    ) * np.log(data_cl)
    # Exclude l = 0
    if l_vec[0] == 0:
        chi2_l = chi2_l[1:]
    chi2 = np.sum(chi2_l)
    loglklhood = -0.5 * chi2
    return loglklhood


# def Kernel_A_of_z_and_f(z, f, A0, zStar):
#     """
#     Returns a parametrized astrophysical kernel as a function of z
#     It does not depend on the frequency
#     """
#     kernel = A0 / 2 * ( np.tanh( 10*(zStar-z) ) + 1 )
#     kernel[kernel < 0.] = 0.
#     return kernel


# def compute_clustering_cl(cosmo, A_z, bias_z, deltaM_k_z, f, k_vec, z_vec, l_vec, jl_fpath=None):
#     """
#     TODO: precompute spherical bessel funcs on a grid of (l , k*chi)
#     TODO: sample differently the spherical Bessel (densely) and deltaM (sparsely)
#     TODO: get rid of the "for" loop over l
#     """

#     # Calculate the Hubble parameter
#     H_z = HubbleZ(cosmo, z_vec)
#     # Create sparse l_array, because C_ell should be relatively smooth at large ell
#     # so we don't need to evaluate it at all ell

#     l_sparse_vec = make_sparse(l_vec)

#     if jl_fpath:
#         # Check if all ell's in l_sparse have been precomputed
#         _l_stored = read_data_from_fits(jl_fpath, 'l')
#         assert set(_l_stored) >= set(l_sparse_vec) , "Some ell's have not been precomputed"

#         # bessel_k_z = read_data_from_fits(jl_fpath, 'bessel_{}'.format(l))
#         hdul = fits.open(jl_fpath)
#     else:
#         _k_vec = k_vec[:,None]
#         _chi_vec = chi_from_z(cosmo,z_vec)[None,:]
#         k_chi_vec = _k_vec * _chi_vec


#     clustering = 0.*l_sparse_vec
#     cosmo_factor = f /( 4.0 * np.pi * rho_crit(cosmo) * constants['cMpcInvSec'] ) * (A_z * bias_z / H_z) * deltaM_k_z
#     for lIndex , l in enumerate(l_sparse_vec):
#         # Note that we use l and not lIndex
#         if jl_fpath:
#             bessel_k_z = hdul['bessel_{}'.format(l)].data
#         else:
#             bessel_k_z = spherical_jn(n=l,z=k_chi_vec)

#         integrand =  cosmo_factor * bessel_k_z

#         Integral = np.empty_like(k_vec)
#         for kIndex, k in enumerate(k_vec):
#             Integral[kIndex] = simps(integrand[kIndex,:],z_vec)

#         clustering[lIndex] = 2.0 / np.pi * simps(np.square(k_vec) * np.square(np.abs(Integral)),k_vec)
#     if jl_fpath:
#         hdul.close()
#     # interpolate the clustering over all ell's
# #     clustering = interp1d(l_sparse_vec, clustering, kind='cubic')(l_vec)

#     return clustering


# # def compute_cosmic_variance(l_vec):
# #     cosmic_variance = 0*l_vec
# #     return cosmic_variance


# def compute_space_shot_noise(cosmo, A_z, f, l_vec, chi_vec):
#     shot_contrib = ( f /rho_crit(cosmo) /constants['cMpcInvSec'] * A_z )**2 / chi_vec**2
#     nBar = 10**(-1) # comoving number density in Mpc-3
#     shot_noise = 1/(4*np.pi)**2 * simps(shot_contrib / nBar, chi_vec)
#     return np.full(len(l_vec),shot_noise)


# def compute_time_shot_noise(l_vec, space_shot_noise, t_obs):
#     # t_obs is the observation time in years
#     beta_T = 1.E-6 * t_obs
#     shot_noise = space_shot_noise / beta_T
#     return np.full(len(l_vec),shot_noise)


# def compute_ifo_noise(l_vec, alpha, n_A, n_B, f, b_AB, t_obs, f_min, f_max,n_per_period=10):
#     # t_obs in years
#     # f_min, f_max in Hz

#     # calculate inv amplitude

#     n_LIGO = 0.8E-47    # in Hz-1
#     f_ref_LIGO = 100.    # in Hz
#     b_LIGO = 3000.    # in km
#     amplitude_inv = 1.5E24 * (1.9*10.)**(-2.*alpha) * (n_LIGO/n_A) * (n_LIGO/n_B) * \
#         (f/f_ref_LIGO)**(2.*alpha) * (b_AB/b_LIGO)**(-2.*alpha+5.) * t_obs

#     # calculate integral contribution

#     beta_min , beta_max = 0.0627 * (b_AB / b_LIGO) * np.array( [f_min, f_max] )
#     # by default have 10 points per period
#     num_points = (beta_max - beta_min) / (2.*np.pi) * n_per_period
#     beta_vec = np.linspace(beta_min, beta_max, num=int(num_points))
#     print(beta_min)
#     print(beta_max)
#     print(int(num_points))
#     integrand = np.zeros((len(l_vec),len(beta_vec)))
#     for l_index, l in enumerate(l_vec):
#         integrand[l_index,:] = beta_vec**(-6. + 2.*alpha) * np.square(abs(spherical_jn(n=l,z=beta_vec) ) )

#     # calculate integral
#     integral = simps(integrand, beta_vec, axis=-1)

#     # combine and invert
#     ifo_noise = (amplitude_inv * integral)**(-1)
#     print("ifo_noise:\n",ifo_noise)

#     return ifo_noise


# def compute_noise_cl(cosmo, A_z, f, l_vec, chi_vec, t_obs):
#     space_shot_noise = compute_space_shot_noise(cosmo, A_z, f, l_vec, chi_vec)
#     time_shot_noise = compute_time_shot_noise(l_vec, space_shot_noise, t_obs)
#     # haven't assigned all the parameters of instrumental noise yet
#     # ifo_noise = compute_ifo_noise(l_vec)
#     ifo_noise = 0 * l_vec

#     noise = space_shot_noise + time_shot_noise + ifo_noise
#     return noise, space_shot_noise, time_shot_noise, ifo_noise


def read_data_from_fits(fname, name):
    """Open a fits file and read data from it.
    Args:
        fname: path of the data file.
        name: name of the data we want to extract.
    Returns:
        array with data for name.
    """
    with fits.open(fname) as fn:
        return fn[name].data


# def read_header_from_fits(fname, name):
#     """ Open a fits file and read header from it.
#     Args:
#         fname: path of the data file.
#         name: name of the data we want to extract.
#     Returns:
#         header.
#     """
#     with fits.open(fname) as fn:
#         return fn[name].header


def write_to_fits(fname, array, name, type="image", header=None):
    """Write an array to a fits file.
    Args:
        fname: path of the input file.
        array: array to save.
        name: name of the image.
    Returns:
        None
    """

    warning = False

    # If file does not exist, create it
    if not os.path.exists(fname):
        hdul = fits.HDUList([fits.PrimaryHDU()])
        hdul.writeto(fname)
    # Open the file
    with fits.open(fname, mode="update") as hdul:
        try:
            hdul.__delitem__(name)
        except KeyError:
            pass
        if type == "image":
            hdul.append(fits.ImageHDU(array, name=name, header=header))
        elif type == "table":
            hdul.append(array)
        else:
            print("Type " + type + " not recognized! Data not saved to file!")
            return True
    print("Appended " + name.upper() + " to " + os.path.relpath(fname))
    sys.stdout.flush()
    return warning


def print_info_fits(fname):
    """Print on screen fits file info.
    Args:
        fname: path of the input file.
    Returns:
        None
    """

    with fits.open(fname) as hdul:
        print(hdul.info())
        sys.stdout.flush()
    return


#     # f = 0.001
#     # A_z = A_z_f_interp(z_vec,f)
#     # clustering_cl = compute_clustering_cl(cosmo, A_z, bias_z, deltaM_k_z, f, k_vec, z_vec, l_vec, path['bessel'])
#     # clustering_cl = clustering_cl[1:]

#     # data_to_compare_path = os.path.join(args.input_dir,'ClGiulia0p001.dat')
#     # data_to_compare_cl_path = os.path.abspath(data_to_compare_path)
#     # data_to_compare_cl = np.loadtxt(data_to_compare_cl_path)
#     # data_to_compare_cl = data_to_compare_cl[:800,1]

#     # import matplotlib.pyplot as plt
#     # plt.loglog(l_vec[1:],clustering_cl,l_vec[1:],data_to_compare_cl)
#     # plt.show()


# def compare_cls(data_cl_path,data_to_compare_path,f_value):
#     # '../data/stochastic_GW/data_cl_f_l.fits'
#     # '/home/yannis/Downloads/ClGiulia63.dat'
#     # data_cl_path  = os.path.abspath(os.path.join(__file__ ,"../..","data/stochastic_GW/data_cl_f_l.fits"))
#     # use: compare_cls('../data/stochastic_GW/data_cl_f_l.fits','/home/yannis/Downloads/ClGiulia63.dat',63.1)
#     import os


#     l_vec = read_data_from_fits(data_cl_path, 'l_vec')
#     f_vec = read_data_from_fits(data_cl_path, 'f_vec')
#     clustering_cl = read_data_from_fits(data_cl_path, 'clustering')
#     noise_cl = read_data_from_fits(data_cl_path, 'noise')

#     clustering_cl = np.array([interp1d(f_vec,clustering_cl[:,ell],kind='cubic') for ell in l_vec])
#     clustering_cl = np.array([clustering_cl[ell](f_value) for ell in l_vec])
#     clustering_cl = clustering_cl[1:]
#     noise_cl = np.array([interp1d(f_vec,noise_cl[:,ell],kind='cubic') for ell in l_vec])
#     noise_cl = np.array([noise_cl[ell](f_value) for ell in l_vec])
#     noise_cl = noise_cl[1:]

#     data_to_compare_cl_path = os.path.abspath(data_to_compare_path)
#     data_to_compare_cl = np.loadtxt(data_to_compare_cl_path)
#     data_to_compare_cl = data_to_compare_cl[:800,1]
#     print(clustering_cl.shape)
#     print(data_to_compare_cl.shape)
#     import matplotlib.pyplot as plt

#     plt.loglog(l_vec[1:],clustering_cl,l_vec[1:],data_to_compare_cl)
#     plt.show()


# def compare_clustering_cls_k(k_min_list):
#     import matplotlib.pyplot as plt

#     f = 63.1
#     data_to_compare_path = os.path.join(args.input_dir,'ClGiulia63.dat')
#     A_z = A_z_f_interp(z_vec,f)

#     clustering_cl_dict = {}
#     for k_min in k_min_list:
#         k_vec = create_array(k_min, args.k_max, args.k_num, args.k_spacing)

#         deltaM_k_z = get_deltaM_array(cosmo, k_vec, z_vec)
#         write_sph_bessel(path['bessel'], make_sparse(l_vec), k_vec, z_vec, chi_vec)

#         clustering_cl_dict[k_min]  = compute_clustering_cl(cosmo, A_z, bias_z, deltaM_k_z, f, k_vec, z_vec, l_vec, path['bessel'])
#         clustering_cl_dict[k_min] = clustering_cl_dict[k_min][1:]
#         plt.loglog(l_vec[1:],clustering_cl_dict[k_min],label=r'$k_{{min}}=$ {}'.format(k_min))

#     data_to_compare_cl_path = os.path.abspath(data_to_compare_path)
#     data_to_compare_cl = np.loadtxt(data_to_compare_cl_path)
#     data_to_compare_cl = data_to_compare_cl[:800,1]
#     plt.loglog(l_vec[1:],data_to_compare_cl)

#     plt.title(r'$C_\ell$ comparison for different $k_{{min}}$ , $f={}$Hz, $k_{{max}}=60$Mpc-1, $\chi_{{min}}=0$Mpc'.format(round(f,3)))
#     plt.xlabel(r'$\ell$')
#     plt.ylabel(r'$C_\ell$')

#     plt.show()


# def compare_clustering_cls_z(z_min_list):
#     import matplotlib.pyplot as plt

#     f = 63.1
#     data_to_compare_path = os.path.join(args.input_dir,'TableClGW_LIGO63.dat')
#     for z_index , z_min in enumerate(z_min_list):
#         data_to_compare_cl_path = os.path.abspath(data_to_compare_path)
#         data_to_compare_cl_dict = np.loadtxt(data_to_compare_cl_path)
#         data_to_compare_cl = data_to_compare_cl_dict[:800,z_index+1]
#         plt.loglog(l_vec[1:], data_to_compare_cl, '--', linewidth=3, label=r'Mathematica $\chi_{{min}}=$ {}'.format(chi_min[z_index]))

#     clustering_cl_dict = {}
#     for z_index , z_min in enumerate(z_min_list):
#         z_vec = create_array(z_min, args.z_max, args.z_num, args.z_spacing)
#         tau_vec = tau_from_z(cosmo,z_vec)
#         chi_vec = chi_from_z(cosmo, z_vec)
#         A_z = A_z_f_interp(z_vec,f)
#         bias_z = biasFactor(args.b0,z_vec)

#         deltaM_k_z = get_deltaM_array(cosmo, k_vec, z_vec)
#         write_sph_bessel(path['bessel'], make_sparse(l_vec), k_vec, z_vec, chi_vec)

#         clustering_cl_dict[z_min]  = compute_clustering_cl(cosmo, A_z, bias_z, deltaM_k_z, f, k_vec, z_vec, l_vec, path['bessel'])
#         clustering_cl_dict[z_min] = clustering_cl_dict[z_min][1:]
#         plt.loglog(l_vec[1:],clustering_cl_dict[z_min],label=r'$\chi_{{min}}=$ {} Mpc'.format(chi_min[z_index]))


#     plt.title(r'$C_\ell$ comparison for different $\chi_{{min}}$ , $f={}$Hz, $k_{{max}}=60$Mpc-1'.format(round(f,3)))
#     plt.xlabel(r'$\ell$')
#     plt.ylabel(r'$C_\ell$')

#     plt.show()

# def generate_noise_table(z_min_list):
#     import matplotlib.pyplot as plt

#     f = 0.001

#     clustering_cl_dict = {}
#     noise_cls = np.zeros(len(z_min_list))
#     for z_index , z_min in enumerate(z_min_list):
#         z_vec = create_array(z_min, args.z_max, args.z_num, args.z_spacing)
#         tau_vec = tau_from_z(cosmo,z_vec)
#         chi_vec = chi_from_z(cosmo, z_vec)
#         A_z = A_z_f_interp(z_vec,f)
#         bias_z = biasFactor(args.b0,z_vec)

#         noise_cls[z_index] = compute_noise_cl(cosmo, A_z, f, l_vec, chi_vec)[0]
#         plt.loglog(l_vec[1:],np.full(len(l_vec[1:]),noise_cls[z_index]),label=r'$Noise \chi_{{min}}=$ {} Mpc'.format(chi_min[z_index]))
#         print('z_min = {} chi_min = {}, Shot Noise = {}'.format(z_min,chi_min[z_index],noise_cls[z_index]))

#         deltaM_k_z = get_deltaM_array(cosmo, k_vec, z_vec)
#         write_sph_bessel(path['bessel'], make_sparse(l_vec), k_vec, z_vec, chi_vec)

#         clustering_cl_dict[z_min]  = compute_clustering_cl(cosmo, A_z, bias_z, deltaM_k_z, f, k_vec, z_vec, l_vec, path['bessel'])
#         clustering_cl_dict[z_min] = clustering_cl_dict[z_min][1:]
#         plt.loglog(l_vec[1:],clustering_cl_dict[z_min],label=r'$Signal \chi_{{min}}=$ {} Mpc'.format(chi_min[z_index]))

#     plt.title(r'$C_\ell$ and $N_\ell$ for different $\chi_{{min}}$ , $f={}$Hz, $k_{{max}}=60$Mpc-1'.format(round(f,4)))
#     plt.xlabel(r'$\ell$')
#     plt.ylabel(r'$C_\ell$')

#     plt.show()

#     import pandas as pd

#     df = pd.DataFrame({"z_min": chi_min, "chi_min": z_min_list, "Shot_Noise": noise_cls})
#     df.to_csv("~/Desktop/noise_cutoffs.csv",'\t',columns=["z_min","chi_min","Shot_Noise"], index=False)


def parser_with_arguments():
    def float_or_Tracer(value):
        try:
            return float(value)
        except ValueError:
            return value

    parser = argparse.ArgumentParser(
        "This is a standalone code that produces the fake stochastic GW  data.\n"
        "       It generates a file\n"
        "       containing a table of the GW Cls for each frequency and ell to be\n"
        "       used in likelihood analysis."
    )

    parser.add_argument("input_dir", type=str, help="Input folder")
    parser.add_argument(
        "--f_fname",
        "-f",
        type=str,
        default="fvec.dat",
        help="File name of array of frequencies",
    )
    parser.add_argument(
        "--z_fname",
        "-z",
        type=str,
        default="zvec.dat",
        help="File name of array of redshifts",
    )
    parser.add_argument(
        "--A_fname",
        "-A",
        type=str,
        default="LumA.dat",
        help="File name of table for the astrophysical kernel",
    )
    parser.add_argument(
        "--output_path", "-o", type=str, default=None, help="Output folder"
    )
    parser.add_argument("--bessel_path", type=str, default=None, help="Bessel folder")

    parser.add_argument("--l_max", type=int, default=800, help="Maximum ell")
    parser.add_argument("--x_min", type=float, default=0.5, help="Minimum k*chi")
    parser.add_argument(
        "--num_after_max",
        type=int,
        default=800,
        help="Number of periods in grid after nu(l_max)",
    )
    parser.add_argument(
        "--points_pp",
        type=int,
        default=10,
        help="Points per period of the spherical Bessel",
    )
    parser.add_argument(
        "--min_after_nu",
        type=int,
        default=50,
        help="Minimum number of periods of the spherical Bessel after nu",
    )
    parser.add_argument(
        "--num_before_nu",
        type=int,
        default=10,
        help="number of periods of the spherical Bessel before nu",
    )

    parser.add_argument(
        "--z_min", type=float, default=0.01356, help="Minimum redshift"
    )  # 0.01356 Corresponds to r_min = 60 Mpc
    parser.add_argument("--z_max", type=float, default=7.0, help="Maximum redshift")

    parser.add_argument("--k_min", type=float, default=1.0e-4, help="Minimum k")
    parser.add_argument("--k_max", type=float, default=60.0, help="Maximum k")
    parser.add_argument(
        "--k_density",
        type=int,
        default=20,
        help="Number of k points per order of magnitude",
    )

    parser.add_argument(
        "--f_num", type=int, default=300, help="Number of frequency points"
    )
    parser.add_argument(
        "--f_spacing",
        type=str,
        default="log",
        choices=["linear", "log"],
        help="Spacing of frequency points",
    )

    parser.add_argument(
        "--k_sparse_num",
        type=int,
        default=60,
        help="Number of k points in the sparse grid",
    )
    parser.add_argument(
        "--z_sparse_num",
        type=int,
        default=50,
        help="Number of z/chi points in the sparse grid",
    )

    # parser.add_argument('--A_max', type=float_or_Tracer,
    #                     default = 6.00E-38,
    #                     help='Amplitude of gaussian kernel')
    # parser.add_argument('--mean_z', type=float_or_Tracer, default = 0.6,
    #                     help='Position of peak of gaussian kernel')
    # parser.add_argument('--sigma_z', type=float_or_Tracer, default = 0.7,
    #                     help='Standard deviation of gaussian kernel')

    parser.add_argument("--b0", type=float, default=1.5, help="Current bias")
    parser.add_argument(
        "--t_obs", type=float, default=1.0, help="Observation time in years"
    )
    parser.add_argument(
        "--n_G",
        type=float,
        default=1e-2,
        help="Comoving number density of galaxies in inverse cubic megaparsecs.",
    )
    parser.add_argument("--preBessel", action="store_true", help="Precompute Bessel")
    parser.add_argument("--storeCl", action="store_true", help="Store C_ell's")
    parser.add_argument(
        "--overwriteKernel",
        action="store_true",
        help="Overwrite the kernel from file with parametrized (Gauss)",
    )
    parser.add_argument(
        "--full_ell",
        action="store_true",
        help="Calculate spherical Bessel and C_ell at all ell between 0 and l_max",
    )
    return parser


def compute_cl(samples, args, f_value=None, f_min=None, f_max=None, verbose=True):
    # samples contains the values of the parameters of the kernel
    # args contains the arguments of the code
    # transfer the samples to args
    args.A_max = samples[0]
    args.mean_z = samples[1]
    args.sigma_z = samples[2]
    nonlinear = "Halofit"
    # Write here cosmological parameters used to calculate the data
    # params_cosmo = {
    #     "output": "mPk",
    #     "z_pk": "0., 3.0, 7.0, 10.0",
    #     "P_k_max_1/Mpc": "70",
    #     "non linear": nonlinear,
    #     #         'gauge' : 'Newtonian' #TODO: commented this as it should be the same. Check!
    # }

    if not args.output_path:  # Assign default name
        if args.overwriteKernel:
            args.output_path = os.path.join(args.input_dir, "data_cl_f_l_GAUSS.fits")
        else:
            args.output_path = os.path.join(args.input_dir, "data_cl_f_l_TABLES.fits")

    if not args.bessel_path:  # Assign default name
        args.bessel_path = os.path.join(
            args.input_dir, "sph_bessel_k_z_l_TEST_gwtools.fits"
        )

    # Assign absolute path for all files
    input_dir = os.path.abspath(args.input_dir)
    path = {
        "f": os.path.join(input_dir, args.f_fname),
        "z": os.path.join(input_dir, args.z_fname),
        "A": os.path.join(input_dir, args.A_fname),
        "output": os.path.abspath(args.output_path),
        "bessel": os.path.abspath(args.bessel_path),
    }

    ### NOTE: temporary comment out. put back in when interpolating from files

    # Check existence of input files and warning before overwriting output
    # assert os.path.isfile(path['f']), 'File {} not found!'.format(path['f'])
    # assert os.path.isfile(path['z']), 'File {} not found!'.format(path['z'])
    # assert os.path.isfile(path['A']), 'File {} not found!'.format(path['A'])
    # if os.path.isfile(path['output']) and args.storeCl:
    #     print('WARNING! I am going to overwrite a pre-existing data file!')

    # # Import files
    # # initially 71 redshifts and 141 frequencies. Kernel A in erg/cm^3.
    # f_vec  = np.genfromtxt(path['f'], delimiter='\t')
    # z_vec = np.genfromtxt(path['z'], delimiter='\t')
    # A_z_f  = np.genfromtxt(path['A'])
    # # Check that the imported arrays have the right dimensions and consistent with input parameters
    # assert z_vec.shape+f_vec.shape==A_z_f.shape, 'The dimensions of the imported arrays are wrong!'
    # assert args.z_min>=z_vec.min() and args.z_max<=z_vec.max(), 'Check redshift boundaries!'
    # # Interpolate the kernel and check that the arguments are z and f in this order
    # A_z_f_interp = interp2d(z_vec, f_vec, A_z_f.T, kind='cubic')
    # assert z_vec.min()==A_z_f_interp.x_min and z_vec.max()==A_z_f_interp.x_max
    # assert f_vec.min()==A_z_f_interp.y_min and f_vec.max()==A_z_f_interp.y_max
    l_vec = np.arange(args.l_max + 1)
    if args.full_ell:
        l_compute = np.arange(args.l_max + 1)
    else:
        l_compute = jax.jit(make_sparse, static_argnums=(0,))(args.l_max)

    ### DENSE

    x_vec = get_x_full(
        ell_max=args.l_max,
        x_min=args.x_min,
        after=args.num_after_max,
        points_pp=args.points_pp,
    )

    # Create vectors
    k_num = int(args.k_density * (np.log10(args.k_max) - np.log10(args.k_min)))
    k_vec = create_array(args.k_min, args.k_max, k_num, "log")

    ### SPARSE
    k_sparse = create_array(args.k_min, args.k_max + 1, args.k_sparse_num, "log")
    z_sparse = create_array(args.z_min, args.z_max, args.z_sparse_num, "log")

    intermediate_grids = get_intermediate_grids(k_vec, x_vec, k_sparse)

    #     print("NOTE: choosing very narrow frequency interval")
    if f_value is None:
        f_vec = create_array(f_min, f_max, args.f_num, args.f_spacing)
    else:
        f_vec = [
            f_value,
        ]

    # Calculate the matter power spectrum. This is frequency independent.
    # This is the only place were we need Class
    # cosmo = Class()
    # cosmo.set(params_cosmo)
    # cosmo.compute()

    cosmo = jc.Planck15()

    b_eff, deltaM_eff, assorted_grids = get_cosmo_eff(
        cosmo, z_sparse, intermediate_grids, args, nonlinear
    )

    # used in the evaluation of noise
    chi_mid = chi_from_z(cosmo, assorted_grids["z_mid"])

    # Precompute Spherical Bessel Function
    try:
        if verbose:
            print("Checking for pre-computed Spherical Bessel")
        assert os.path.isfile(path["bessel"])
        _l = read_data_from_fits(path["bessel"], "l")
        _x = read_data_from_fits(path["bessel"], "x")
        assert _x.size == x_vec.size
        assert _x.min() == x_vec.min()
        assert _x.max() == x_vec.max()
        # TODO: re-implement this check in jax
        # assert set(_l) >= set(l_compute)
    except AssertionError:
        print("Could not find consistent precomputed Bessel")
        if args.preBessel:
            print("WARNING! I am going to overwrite the precomputed bessel file!")
            print(
                "Writing Bessel. This might take a while time and it might require a lot of memory"
            )

            def after_func(ell):
                return max(ell, args.min_after_nu)

            write_sph_bessel(
                path["bessel"],
                l_compute,
                x_vec=x_vec,
                before=args.num_before_nu,
                after=after_func,
            )
            print("Finished writing Bessel")
        else:
            path["bessel"] = None
            print("Not going to use precomputed Bessel Function")
            print(
                "Use --preBessel True to store and use spherical Bessel Functions for these k and z vectors"
            )
    else:
        if verbose:
            print("Found Precomputed Bessels")
    if verbose:
        print("Recovering Bessel")
    bessel_x_l = get_bessel_x_l(path["bessel"], l_compute)[None, ...]

    # Compute clustering
    clustering = np.zeros((len(f_vec), len(l_vec)))
    noise = np.zeros(len(f_vec))
    data = np.zeros((len(f_vec), len(l_vec)))
    for nf, f in enumerate(f_vec):
        print(f"\r{nf} {f:.4f}\tHz", end=" ")
        sys.stdout.flush()

        if args.overwriteKernel:
            A_eff, A_sparse = compute_kernel_on_grid(
                samples,
                freq=f,
                assorted_grids=assorted_grids,
                args=args,
                A_kernel_interp2d=None,
            )
        # else:
        #     A_eff, A_sparse = compute_kernel_on_grid(
        #         freq=f,
        #         assorted_grids=assorted_grids,
        #         args=args,
        #         A_kernel_interp2d=A_z_f_interp,
        #     )

        noise = noise.at[nf].set(
            compute_spatial_shot_noise(
                cosmo, A_z=A_sparse, chi_vec=chi_mid, n_G=args.n_G
            )
        )

        clustering_l = compute_clustering_cl(
            cosmo,
            f,
            A_eff,
            b_eff,
            deltaM_eff,
            bessel_x_l,
            intermediate_grids["chi_grid"],
            k_vec,
        )

        # clustering[nf,:] = interpolate_cl(clustering_l,
        #                                          l_compute,
        #                                          l_vec,
        #                                          "linear")
        clustering = clustering.at[nf].set(
            interpolate_cl(
                clustering_l,
                l_compute,
                l_vec,
            )
        )

        # data[nf,:] = clustering[nf,:] + noise[nf]
        data = data.at[nf].set(clustering[nf, :] + noise[nf])

    return data, clustering, noise, l_vec, f_vec


# ---------------------------- Main ------------------------------------------ #

if __name__ == "__main__":
    """
    This is a standalone code that produces the fake stochastic GW
    data. It is completely independent of MontePython. Given an input
    folder containing three files, i.e. array of frequencies, array of
    redshifts and table with the astrophysical kernel, it generates
    a file containing the GW Cl's (both signal and noise) for each
    frequency and ell.
    Args:
        - input_dir (obl): folder where the three files are stored
        - f_fname (opt, default: fvec.dat): file name of array of frequencies
        - z_fname (opt, default: zvec.dat): file name of array of redshifts
        - A_fname (opt, default: LumA_Mod0.dat): file name of astrophysical table
        - output_path (opt, default: [input_dir]/data_cl_f_l.fits): output file
    """

    # Here we import only those modules that are necessary here but not on the
    # likelihood analysis. This way we minimize the requested import on normal runs
    # TODO: for now I am writing all modules here. When a module is used above it should be deleted from here

    # Parse the command line arguments

    parser = parser_with_arguments()

    args = parser.parse_args()
    f_min = 63.05
    f_max = 63.15
    f_value = 63.1

    if args.storeCl:
        data, clustering, noise, l_vec, f_vec = compute_cl(
            args, f_min=f_min, f_max=f_max
        )

        input_dir = os.path.abspath(args.input_dir)
        path = {
            "f": os.path.join(input_dir, args.f_fname),
            "z": os.path.join(input_dir, args.z_fname),
            "A": os.path.join(input_dir, args.A_fname),
            "output": os.path.abspath(args.output_path),
            "bessel": os.path.abspath(args.bessel_path),
        }

        print("")

        # Save data to output file
        write_to_fits(path["output"], l_vec, "l_vec", type="image", header=None)
        write_to_fits(path["output"], f_vec, "f_vec", type="image", header=None)
        write_to_fits(path["output"], data, "data", type="image", header=None)
        write_to_fits(
            path["output"], clustering, "clustering", type="image", header=None
        )
        write_to_fits(path["output"], noise, "noise", type="image", header=None)
        #         write_to_fits(path['output'], space_shot_noise, 'noise_shot_s', type='image', header=None)
        #         write_to_fits(path['output'], time_shot_noise, 'noise_shot_t', type='image', header=None)
        #         write_to_fits(path['output'], ifo_noise, 'noise_ifo', type='image', header=None)

        print_info_fits(path["output"])
    else:
        data, clustering, noise, l_vec, f_vec = compute_cl(args, f_value=f_value)
