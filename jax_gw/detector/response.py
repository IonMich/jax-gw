import jax.numpy as jnp

from jax_gw.detector.orbits import flat_index, get_arm_lengths, path_from_indices

C_IN_AU_PER_S = 0.0020039888


def antenna_pattern(
    u_hat: jnp.array, v_hat: jnp.array, arm_direction: jnp.array
) -> jnp.array:
    """Calculate the antenna pattern for a given source direction.

    Parameters
    -------
    u_hat : jnp.array
        First unit vector in the transverse plane of the incoming signal.
    v_hat : jnp.array
        Second unit vector in the transverse plane of the incoming signal.
    arm_direction : jnp.array
        Unit vector pointing along the arm from the emitter to the receiver.

    Returns
    -------
    jnp.array
        Plus and cross antenna pattern functions.
    """
    ksi_plus = jnp.dot(arm_direction, u_hat) ** 2 - jnp.dot(arm_direction, v_hat) ** 2
    ksi_cross = 2 * jnp.dot(arm_direction, u_hat) * jnp.dot(arm_direction, v_hat)

    return jnp.stack([ksi_plus, ksi_cross], axis=-1)


def transfer_function(
    k_hat: jnp.array,
    freq: jnp.array,
    arms: jnp.array,
) -> jnp.array:
    """Calculate the transfer function for a given source direction.

    Parameters
    ----------
    k_hat : jnp.array
        Unit vector pointing in the direction of propagation of the incoming
        signal.
    freq : jnp.array
        Frequencies of the gravitational wave.
    arms : jnp.array
        Arm configuration of the spacecraft.

    Returns
    -------
    jnp.array
        Transfer function for the given source direction.
    """
    arm_length = get_arm_lengths(arms)

    delta_t = arm_length - jnp.dot(arms, k_hat)
    delta_t = delta_t / C_IN_AU_PER_S
    # jnp outer flattens the array, so we need to reshape it
    freq = jnp.atleast_1d(freq)
    delta_phi = jnp.pi * jnp.outer(freq, delta_t).reshape(freq.shape + delta_t.shape)

    return jnp.sinc(delta_phi / jnp.pi) * jnp.exp(-1j * delta_phi)


def response_function(
    k_hat: jnp.array,
    freq: jnp.array,
    receiver_positions: jnp.array,
    full_transfer: jnp.array,
    antennae: jnp.array,
) -> jnp.array:
    """Calculate the response function for a given source direction.

    Parameters
    ----------
    k_hat : jnp.array
        Unit vector pointing in the direction of propagation of the incoming
        signal.
    freq : jnp.array
        Frequencies of the gravitational wave.
    receiver_positions : jnp.array
        Positions of the receivers.
    full_transfer : jnp.array
        Transfer function for the given source direction.
        Shape: (N_sky, N_freq, N_pair, N_times)
    antennae : jnp.array
        Plus and cross antenna pattern functions.
        Shape: (N_sky, N_pair, N_times, N_pol)

    Returns
    -------
    jnp.array
        Response function for the given source direction.
    """

    dot_product = jnp.dot(receiver_positions, k_hat) / C_IN_AU_PER_S
    delta_phi = (
        2
        * jnp.pi
        * jnp.outer(freq, dot_product).reshape(freq.shape + dot_product.shape)
    )
    delta_phi = jnp.moveaxis(delta_phi, -1, 0)
    position_phase_shift = jnp.exp(-1j * delta_phi)

    # add the position phase shift to the transfer function
    full_transfer = full_transfer * position_phase_shift

    # response function assuming no time delay
    response_no_delay = 0.5 * jnp.einsum("ij...,i...k->...ijk", full_transfer, antennae)
    return response_no_delay


def get_path_response(
    paths: jnp.array,
    freqs: jnp.array,
    arm_lengths: jnp.array,
    response: jnp.array,
):
    """Calculate the response function for a collection of photon paths.

    Parameters
    ----------
    paths : jnp.array
        Spacecraft indices for photon paths in shape (N_paths, N_depth).
    freqs : jnp.array
        Frequencies of the gravitational wave in shape (N_freq,).
    arm_lengths : jnp.array
        Arm lengths of the spacecraft in shape (N_pair, N_times).
    response : jnp.array
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

    path_separations = arm_lengths[flat_indices]
    # no time delay for the first element of each path
    path_separations = jnp.insert(path_separations, 0, 0.0, axis=1)
    path_separations = path_separations[:, :-1, ...]
    path_separations = jnp.moveaxis(path_separations, -1, 0)

    cumul_path_separations = jnp.cumsum(path_separations, axis=-1)
    cumul_path_phases = -2 * jnp.pi * jnp.outer(freqs, cumul_path_separations)
    cumul_path_phases = (
        cumul_path_phases.reshape(freqs.shape + cumul_path_separations.shape)
        / C_IN_AU_PER_S
    )
    cumul_path_exp = jnp.exp(1j * cumul_path_phases)

    path_responses = jnp.einsum(
        "ijkl,kljmin->kjmin", cumul_path_exp, response[flat_indices]
    )

    return path_responses
