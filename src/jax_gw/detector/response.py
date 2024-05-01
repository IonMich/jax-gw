import jax
from jax import Array
from jax.typing import ArrayLike

import jax.numpy as jnp

from jax_gw.detector.orbits import (
    flat_index,
    flatten_pairs,
    get_arm_lengths,
    get_receiver_positions,
    get_separations,
    path_from_indices,
)

C_IN_AU_PER_S = 0.0020039888


def antenna_pattern(
    u_hat: ArrayLike,
    v_hat: ArrayLike,
    arm_direction: ArrayLike,
) -> Array:
    """Calculate the antenna pattern for a given source direction.

    Parameters
    -------
    u_hat : ArrayLike
        First unit vector in the transverse plane of the incoming signal.
    v_hat : ArrayLike
        Second unit vector in the transverse plane of the incoming signal.
    arm_direction : ArrayLike
        Unit vector pointing along the arm from the emitter to the receiver.

    Returns
    -------
    jnp.array
        Plus and cross antenna pattern functions.
    """
    ksi_plus = jnp.dot(arm_direction, u_hat) ** 2 - jnp.dot(arm_direction, v_hat) ** 2
    ksi_cross = 2 * jnp.dot(arm_direction, u_hat) * jnp.dot(arm_direction, v_hat)

    return jnp.stack([ksi_plus, ksi_cross], axis=-1)


sky_vmapped_antenna_pattern = jax.vmap(
    antenna_pattern, in_axes=(0, 0, None), out_axes=0
)


def transfer_function(
    k_hat: ArrayLike,
    freq: ArrayLike,
    arms: ArrayLike,
) -> Array:
    """Calculate the transfer function for a given source direction.

    Parameters
    ----------
    k_hat : ArrayLike
        Unit vector pointing in the direction of propagation of the incoming
        signal.
    freq : ArrayLike
        Frequencies of the gravitational wave.
    arms : ArrayLike
        Arm configuration of the spacecraft.

    Returns
    -------
    jnp.array
        Transfer function for the given source direction.
    """
    if not (isinstance(freq, (jnp.ndarray, Array))) or jnp.isscalar(freq):
        raise TypeError(f"freq must be an array, got {type(freq)}")

    arm_length = get_arm_lengths(arms)

    delta_t = arm_length - jnp.dot(arms, k_hat)
    delta_t = delta_t / C_IN_AU_PER_S
    L_over_c = arm_length / C_IN_AU_PER_S
    # jnp outer flattens the array, so we need to reshape it
    delta_phi = jnp.pi * jnp.outer(freq, delta_t).reshape(freq.shape + delta_t.shape)

    return L_over_c * jnp.sinc(delta_phi / jnp.pi) * jnp.exp(-1j * delta_phi)


vmapped_transfer_function = jax.vmap(
    transfer_function,
    in_axes=(0, None, None),
    out_axes=0,
)

jitted_vmapped_transfer_function = jax.jit(vmapped_transfer_function)


def response_function(
    k_hat: ArrayLike,
    freq: ArrayLike,
    receiver_positions: ArrayLike,
    full_transfer: ArrayLike,
    antennae: ArrayLike,
) -> Array:
    """Calculate the timing response function for a given source direction.

    Parameters
    ----------
    k_hat : ArrayLike
        Unit vector pointing in the direction of propagation of the incoming
        signal.
    freq : ArrayLike
        Frequencies of the gravitational wave.
    receiver_positions : ArrayLike
        Positions of the receivers.
    full_transfer : ArrayLike
        Transfer function for the given source direction.
        Shape: (N_sky, N_freq, N_pair, N_times)
    antennae : ArrayLike
        Plus and cross antenna pattern functions.
        Shape: (N_sky, N_pair, N_times, N_pol)

    Returns
    -------
    jnp.array
        Response function for the given source direction.
    """
    if not (isinstance(freq, (jnp.ndarray, Array))) or jnp.isscalar(freq):
        raise TypeError(f"freq must be an array, got {type(freq)}")

    dot_product = jnp.dot(receiver_positions, k_hat) / C_IN_AU_PER_S
    delta_phi = (
        2
        * jnp.pi
        * jnp.outer(freq, dot_product).reshape(freq.shape + dot_product.shape)
    )
    delta_phi = jnp.moveaxis(delta_phi, -1, 0)
    position_phase_shift = jnp.exp(-1j * delta_phi)

    # include the position phase shift to the transfer function
    full_transfer = full_transfer * position_phase_shift

    # response function assuming no time delay
    response_no_delay = 0.5 * jnp.einsum("ij...,i...k->...ijk", full_transfer, antennae)
    return response_no_delay


def response_pipe(
    orbits,
    freqs,
    sky_basis,
):
    """Calculate the response function for a given source direction."""
    k_hat, u_hat, v_hat = sky_basis
    separations = get_separations(orbits)

    receiver_orbits = get_receiver_positions(orbits)
    receiver_positions = flatten_pairs(receiver_orbits)

    arms = flatten_pairs(separations)
    arm_lengths = get_arm_lengths(arms)
    arm_directions = arms / arm_lengths[..., None]

    full_transfer = jitted_vmapped_transfer_function(k_hat, freqs, arms)
    antennae = sky_vmapped_antenna_pattern(u_hat, v_hat, arm_directions)

    response = response_function(
        k_hat.T,
        freqs,
        receiver_positions,
        full_transfer,
        antennae,
    )

    return response, antennae


def get_cumulative_path_separations(
    flat_indices: ArrayLike,
    arm_lengths: ArrayLike,
):
    path_separations = arm_lengths[flat_indices]
    # no time delay for the first element of each path
    path_separations = jnp.insert(path_separations, 0, 0.0, axis=1)
    path_separations = jnp.moveaxis(path_separations, -1, 0)

    cumul_path_separations = jnp.cumsum(path_separations, axis=-1)

    return cumul_path_separations


def get_path_response(
    paths: ArrayLike,
    freqs: ArrayLike,
    arm_lengths: ArrayLike,
    response: ArrayLike,
):
    """Calculate the timing response function for a collection of photon paths.

    Parameters
    ----------
    paths : ArrayLike
        Spacecraft indices for photon paths in shape (N_paths, N_depth).
    freqs : ArrayLike
        Frequencies of the gravitational wave in shape (N_freq,).
    arm_lengths : ArrayLike
        Arm lengths of the spacecraft in shape (N_pair, N_times).
    response : ArrayLike
        Response function for the given source direction

    Returns
    -------
    jnp.array
        Response function for the given source direction
    """
    indices = path_from_indices(paths)
    N_pair = response.shape[0]
    # N_pair = N * (N - 1), thus
    N = round(jnp.sqrt(N_pair + 1 / 4) + 1 / 2)

    flat_indices = jnp.apply_along_axis(
        lambda indices: flat_index(*indices, N),
        axis=-1,
        arr=indices,
    )
    # print(flat_indices)

    cumul_path_separations = get_cumulative_path_separations(flat_indices, arm_lengths)
    # remove the last element of each path, as it does not appear in emitter phases
    reduced_cumul_path_separations = cumul_path_separations[..., :-1]

    cumul_path_phases = -2 * jnp.pi * jnp.outer(freqs, reduced_cumul_path_separations)
    cumul_path_phases = (
        cumul_path_phases.reshape(freqs.shape + reduced_cumul_path_separations.shape)
        / C_IN_AU_PER_S
    )
    cumul_path_exp = jnp.exp(1j * cumul_path_phases)

    path_responses = jnp.einsum(
        "ijkl,kljmin->kjmin", cumul_path_exp, response[flat_indices]
    )

    return path_responses, cumul_path_separations


def get_differential_strain_response(
    path_response: ArrayLike,
    path_idx_1: int,
    path_idx_2: int,
    cumul_path_separations: ArrayLike,
):
    """Calculate the strain response from the difference in the responses of two photon paths,
    of equal cumulative length, i.e.

    `R_{diff} = (R[path_idx_1] - R[path_idx_2]) / ( L_tot / c)`

    Parameters
    ----------
    path_response : ArrayLike
        Timing response function for a collection of photon paths.
    path_idx_1 : int
        Index of the first photon path.
    path_idx_2 : int
        Index of the second photon path.
    cumul_path_separations : ArrayLike
        Cumulative path lengths for the photon paths.

    Returns
    -------
    jnp.array
        Michelson strain response for the two chosen photon paths.
    """
    # get the cumulative path lengths for the two paths
    total_length_1 = cumul_path_separations[:, path_idx_1, -1]
    total_length_2 = cumul_path_separations[:, path_idx_2, -1]
    total_length = 0.5 * (total_length_1 + total_length_2)
    total_time = total_length / C_IN_AU_PER_S

    # get the difference in the response functions
    path_response_diff = path_response[path_idx_1] - path_response[path_idx_2]

    # get the strain response
    strain_response = path_response_diff / total_time[:, None, None, None]
    return strain_response


def get_pairwise_differential_strain_response(
    path_response: ArrayLike,
    cumul_path_separations: ArrayLike,
):
    """Calculate the strain response from the difference in the responses of two subsequent photon paths,
    of equal cumulative length, i.e.

    `R_{diff} = (R[path_idx_1] - R[path_idx_2]) / ( L_tot / c)`

    where `path_idx_1` and `path_idx_2` are even and odd successive indices.

    Parameters
    ----------
    path_response : ArrayLike
        Timing response function for a collection of photon paths.
    cumul_path_separations : ArrayLike
        Cumulative path lengths for the photon paths.

    Returns
    -------
    jnp.array
        Michelson strain response for all successive photon pairs.
    """
    total_length_cw = cumul_path_separations[:, ::2, -1]
    total_length_ccw = cumul_path_separations[:, 1::2, -1]
    total_length = 0.5 * (total_length_cw + total_length_ccw)
    total_time = total_length / C_IN_AU_PER_S

    path_response_diff = path_response[::2] - path_response[1::2]

    strain_response = path_response_diff / total_time.T[..., None, None, None]

    return strain_response


def create_cyclic_permutation_paths(path, N):
    path_prime = (N - path) % N

    # N-1 cyclic permutations of (path, path_prime) by adding 1 modulo N
    paths_clockwise = jnp.mod(path + jnp.arange(N)[..., jnp.newaxis], N)
    paths_counter_clockwise = jnp.mod(path_prime + jnp.arange(N)[..., jnp.newaxis], N)
    interleaved_array_shape = (2 * paths_clockwise.shape[0], paths_clockwise.shape[1])
    interleaved_array = jnp.zeros(interleaved_array_shape, dtype=path.dtype)
    interleaved_array = interleaved_array.at[::2].set(paths_clockwise)
    interleaved_array = interleaved_array.at[1::2].set(paths_counter_clockwise)
    return interleaved_array


def get_LISA_A_channel_response():
    pass
