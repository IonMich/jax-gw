# samples contains the values of the parameters of the kernel
# args contains the arguments of the code
# transfer the samples to args
import os
from typing import Union
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_median

from jax_gw.signal.agwb import (
    compute_clustering_cl,
    compute_kernel_on_grid,
    compute_spatial_shot_noise,
    interpolate_cl,
    parser_with_arguments,
    read_data_from_fits,
    write_sph_bessel,
    get_x_full,
    make_sparse,
    create_array,
    get_intermediate_grids,
    get_cosmo_eff,
    get_bessel_x_l,
    chi_from_z,
    compute_cl,
)
from numpy import save as np_save


parser = parser_with_arguments()
args = parser.parse_args(
    "./src/jax_gw/data/stochastic_GW/ --preBessel --overwriteKernel".split()
)

# f=1E-1 Hz fit. power-law in frequency up to f_max = 1E1 Hz.
A_max = 0.510579e-36
mean_z = 0.5784058
sigma_z = 0.6766768

samples = jnp.array([A_max, mean_z, sigma_z])
f_value: Union[float, None] = None
f_ref = 1e-1
f_min = 1e-2
f_max = 1e1
verbose = True


args.A_max = samples[0]
args.mean_z = samples[1]
args.sigma_z = samples[2]
nonlinear = "Halofit"
# Write here cosmological parameters used to calculate the data
params_cosmo = {
    "output": "mPk",
    "z_pk": "0., 3.0, 7.0, 10.0",
    "P_k_max_1/Mpc": "70",
    "non linear": nonlinear,
    #         'gauge' : 'Newtonian' #TODO: commented this as it should be the same. Check!
}

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


l_vec = jnp.arange(args.l_max + 1)
if args.full_ell:
    l_compute = jnp.arange(args.l_max + 1)
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
k_num = int(args.k_density * (jnp.log10(args.k_max) - jnp.log10(args.k_min)))
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

not_chi_mask_nonzero = (~assorted_grids["chi_mask"]).nonzero()


def cl_broadband_from_grids(
    params,
    f_vec,
    f_ref,
    args,
):
    f_len = 1
    clustering = jnp.zeros((f_len, len(l_vec)))
    noise = jnp.zeros(f_len)
    data = jnp.zeros((f_len, len(l_vec)))
    # print(f"\r{0} {f_vec[0]:.4f}-{f_vec[-1]:.4f}\tHz", end=" ", flush=True)
    nf = 0
    A_eff, A_sparse = compute_kernel_on_grid(
        params,
        freq=f_ref,
        assorted_grids=assorted_grids,
        args=args,
        not_chi_mask_nonzero=not_chi_mask_nonzero,
        A_kernel_interp2d=None,
    )
    f_slope = 2 / 3
    broadband = (
        1
        / (f_slope + 2)
        / f_ref**f_slope
        * (f_vec[-1] ** (f_slope + 2) - f_vec[0] ** (f_slope + 2))
    )
    A_eff = A_eff * broadband
    A_sparse = A_sparse * broadband

    noise = noise.at[0].set(
        compute_spatial_shot_noise(cosmo, A_z=A_sparse, chi_vec=chi_mid, n_G=args.n_G)
    )
    clustering_l = compute_clustering_cl(
        cosmo,
        A_eff,
        b_eff,
        deltaM_eff,
        bessel_x_l,
        intermediate_grids["chi_grid"],
        k_vec,
    )
    clustering = clustering.at[nf].set(
        interpolate_cl(
            clustering_l,
            l_compute,
            l_vec,
        )
    )

    # data[nf,:] = clustering[nf,:] + noise[nf]
    data = data.at[nf].set(clustering[nf, :] + noise[nf])
    return data


def generate_cl_data(A_max, z_peak, z_sigma):
    samples = jnp.array([A_max, z_peak, z_sigma])
    args_data = parser.parse_args(
        "./jax_gw/data/stochastic_GW/ --preBessel --overwriteKernel".split()
    )
    cls = compute_cl(samples, args_data, f_value=63.1)
    # add noise
    cls_01 = cls[1][0] + cls[1][1]
    return cls_01


def compute_loglkl_from_cls(data_cl, theory_cl, l_vec):
    # Combining theory_cl and data_cl to calculate the likelihood
    # Using equation (3) of arXiv 1811.11584
    # Note that most of the expression below can be precomputed if needed
    # Note that data_cl already have noise inside

    chi2_l = (
        (2.0 * l_vec + 1.0) * ((data_cl / theory_cl) + jnp.log(theory_cl))
        - (2.0 * l_vec - 1.0) * jnp.log(data_cl)
        - 2.0 * jnp.log(data_cl)
    )

    # Exclude l = 0 from the sum
    chi2_l = chi2_l[1:]
    chi2 = jnp.sum(chi2_l)
    loglklhood = -0.5 * chi2 + 1225505.9
    return loglklhood


def compute_normal_loglkl_from_cls(data_cl, theory_cl, noise_cl, l_vec):
    # Combining theory_cl and data_cl to calculate the likelihood
    # Using a Gaussian and calculating the covariance matrix
    # Note that most of the expression below can be precomputed if needed
    # Note that data_cl already have noise inside

    chi2_l = (2.0 * l_vec + 1.0) / 2 * (
        (data_cl - theory_cl) ** 2 / (theory_cl + noise_cl) ** 2
    ) + 4 / (2 * l_vec + 1) * jnp.log(theory_cl + noise_cl)

    # Exclude l = 0 from the sum
    chi2_l = chi2_l[1:]
    chi2 = jnp.sum(chi2_l)
    loglklhood = -0.5 * chi2
    return loglklhood


def likelihood_fn(A_max=None, z_peak=None, z_sigma=None):
    """Likelihood function for the astrpphysical GW stochastic background (AGWB)

    The likelihood is a Wishart distribution with a covariance given by the AGWB power spectrum

    Args:
        A_max (float): Maximum amplitude of the AGWB
        z_peak (float): Redshift of the peak of the AGWB
        z_sigma (float): Width of the AGWB
    """
    # Sample the parameters
    # with handlers.seed(rng_seed=0):
    A_max_sample = numpyro.sample("A_max", dist.Uniform(0.58, 0.62))
    z_peak_sample = numpyro.sample("z_peak", dist.Uniform(0.48, 0.52))
    z_sigma_sample = numpyro.sample("z_sigma", dist.Uniform(0.58, 0.62))

    str_formatted = "./jax_gw/data/stochastic_GW/ --preBessel --overwriteKernel"
    str_formatted_splitted = str_formatted.split()
    args_data = parser.parse_args(str_formatted_splitted)

    cls = cl_broadband_from_grids(
        jnp.array([A_max_sample * 1e-37, z_peak_sample, z_sigma_sample]),
        f_vec=f_vec,
        f_ref=f_ref,
        args=args_data,
    )
    ell_arr = jnp.arange(0, len(cls[0]))

    # Compute the log likelihood
    loglklhood = compute_loglkl_from_cls(cl_data, cls[0], ell_arr)
    print(loglklhood)
    numpyro.factor("loglklhood", loglklhood)


cl_data = generate_cl_data(6e-38, 0.5, 0.6)

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
kernel = NUTS(
    likelihood_fn,
    init_strategy=init_to_median(),
)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=1)
mcmc.run(rng_key=rng_key_)
mcmc.print_summary()
# store samples
samples_ = mcmc.get_samples()
output = "samples.npy"

np_save(output, samples_)
