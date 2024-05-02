import jax.numpy as jnp

from jax_gw.detector.orbits import (
    earthbound_ifo_pipeline,
    EARTH_Z_LAT,
    EARTH_Z_LON,
    get_receiver_positions,
    flatten_pairs,
    get_separations,
    get_arm_lengths,
)
from jax_gw.detector.pixel import (
    flat_to_matrix_sky_indices,
    get_directional_basis,
    unflatten_sky_axis,
)
from jax_gw.detector.response import (
    C_IN_AU_PER_S,
    antenna_pattern,
    sky_vmapped_antenna_pattern,
    transfer_function,
    response_function,
    response_pipe,
    get_path_response,
    get_differential_strain_response,
    get_pairwise_differential_strain_response,
    create_cyclic_permutation_paths,
)
import pytest

from jax import config

config.update("jax_enable_x64", True)


def test_small_antenna_limit_vs_analytic():
    N_theta = 100
    N_phi = 120
    delta_phi = 2 * jnp.pi / N_phi
    ecl_thetas_reduced = jnp.linspace(1 / N_theta, jnp.pi - 1 / N_theta, N_theta)
    ecl_phis_reduced = jnp.arange(0, 2 * jnp.pi, delta_phi)[:N_phi]

    flat_to_m_sky = flat_to_matrix_sky_indices(N_theta, N_phi)
    ecl_thetas = ecl_thetas_reduced[flat_to_m_sky[:, 0]]
    ecl_phis = ecl_phis_reduced[flat_to_m_sky[:, 1]]
    sky_basis = get_directional_basis(ecl_thetas, ecl_phis)
    _, u_hat, v_hat = sky_basis
    FREQ_ROTATION = 365.25  # in 1/year
    N_times = 4
    times = jnp.linspace(0, 1 / FREQ_ROTATION, N_times)
    r = 1  # in AU
    L_arm = 4  # in km
    orbits = earthbound_ifo_pipeline(
        EARTH_Z_LAT,
        EARTH_Z_LON,
        times,
        r,
        L_arm,
    )

    separations = get_separations(orbits)
    arms = flatten_pairs(separations)
    arm_lengths = get_arm_lengths(arms)
    arm_directions = arms / arm_lengths[..., None]
    antennae = sky_vmapped_antenna_pattern(u_hat, v_hat, arm_directions)
    response_sa = 0.5 * (
        antennae[:, 0] + antennae[:, 1] - antennae[:, 2] - antennae[:, 3]
    )
    response_sa = unflatten_sky_axis(response_sa, axis=0, N_theta=N_theta, N_phi=N_phi)
    response_plus = response_sa[..., 0]

    u = ecl_phis_reduced
    v = ecl_thetas_reduced
    analytic_response = (jnp.cos(u[None, :]) ** 2 - jnp.sin(u[None, :]) ** 2) * (
        jnp.cos(v[:, None]) ** 2 + 1
    )
    f_star = C_IN_AU_PER_S / arm_lengths[0, 0]
    N_freqs = 5
    freqs = jnp.linspace(0, 0.5 * f_star, N_freqs)
    _, antennae_from_pipe = response_pipe(
        orbits,
        freqs,
        sky_basis=sky_basis,
    )
    assert jnp.allclose(antennae_from_pipe, antennae, atol=1e-6)
    assert jnp.allclose(response_plus[..., 0], analytic_response, atol=1e-6)
