import jax.numpy as jnp

from jax_gw.detector.orbits import (
    create_cartwheel_orbit,
    get_separations,
    get_arm_lengths,
    flatten_pairs,
)

from jax_gw.detector.pixel import (
    get_directional_basis,
    flat_to_matrix_sky_indices,
    get_sph_harm_values,
    pixel_to_lm,
)

from jax_gw.detector.response import (
    create_cyclic_permutation_paths,
    response_pipe,
    get_path_response,
    get_pairwise_differential_strain_response,
)

from jax_gw.detector.overlap import (
    unpolarized_cross_overlap,
    overlap_angular_noise_ell,
)


def get_N_ell_BBO(
    N_times=4,
    N_freqs=64,
    N_theta=300,
    N_phi=40,
    l_max=10,
    t_obs=3.16e-5,
    spectral_indices=[-2.3, 0, -3],
):
    f_min = 1e-2
    f_max = 1e1
    L_target = 0.05
    R_target = 1.0
    N = 3

    path_1 = jnp.array(
        [
            0,
            1,
            0,
        ]
    )

    f_ref = 1e-1
    delta_x_sq = 2e-34  # m^2/Hz
    delta_a_sq = 9e-34
    t_obs_noise = 4.0

    times = jnp.linspace(0, t_obs, N_times)
    freqs = jnp.logspace(jnp.log10(f_min), jnp.log10(f_max), N_freqs)

    AU_per_billion_meters = 149.597871

    delta_phi = 2 * jnp.pi / N_phi
    ecl_thetas_reduced = jnp.linspace(1 / N_theta, jnp.pi - 1 / N_theta, N_theta)
    ecl_phis_reduced = jnp.arange(0, 2 * jnp.pi, delta_phi)[:N_phi]
    # print(max(ecl_phis_reduced), min(ecl_phis_reduced))

    flat_to_m_sky = flat_to_matrix_sky_indices(N_theta, N_phi)
    ecl_thetas = ecl_thetas_reduced[flat_to_m_sky[:, 0]]
    ecl_phis = ecl_phis_reduced[flat_to_m_sky[:, 1]]
    sky_basis = get_directional_basis(ecl_thetas, ecl_phis)
    timeshifts = [0, jnp.pi]
    ecc = L_target / (AU_per_billion_meters * 2 * jnp.sqrt(3) * R_target)
    ecc = ecc.item()
    paths_triplet = create_cyclic_permutation_paths(path_1, N)
    orbits_list = []
    for timeshift in timeshifts:
        orbits = create_cartwheel_orbit(ecc, R_target, N, times, timeshift=timeshift)
        orbits_list.append(orbits)

    michelson_responses_pixel = []
    for orbits in orbits_list:
        michelson = get_michelson_triplet(
            orbits,
            freqs,
            sky_basis,
            paths_triplet,
        )
        michelson_responses_pixel.append(michelson)

    michelson_1 = michelson_responses_pixel[0][0]
    michelson_2 = michelson_responses_pixel[1][0]
    anisotropic_ORF_BBO_12 = unpolarized_cross_overlap(michelson_1, michelson_2)
    sph_harm_values = get_sph_harm_values(l_max, ecl_thetas_reduced, ecl_phis_reduced)

    gamma_BBO_12_lm = pixel_to_lm(
        anisotropic_ORF_BBO_12,
        1,
        N_theta,
        N_phi,
        ecl_thetas=ecl_thetas,
        ecl_phis=ecl_phis,
        sph_harm_values=sph_harm_values,
    )

    L_SI = L_target * 1e9
    noise_psd_bbo = (
        4 / (L_SI) ** 2 * (delta_x_sq + (delta_a_sq / (2 * jnp.pi * freqs) ** 4))
    )

    ell_array = jnp.arange(0, l_max + 1)

    BBO_N_ell_alphas = jnp.zeros((len(spectral_indices), l_max + 1))
    for idx, spectral_index in enumerate(spectral_indices):
        BBO_N_ell_alpha = overlap_angular_noise_ell(
            gamma_BBO_12_lm,
            noise_psd_bbo,
            ell_array,
            freqs,
            f_ref,
            spectral_index=spectral_index,
            t_obs=t_obs_noise,
        )
        BBO_N_ell_alphas = BBO_N_ell_alphas.at[idx].set(BBO_N_ell_alpha)

    return BBO_N_ell_alphas


def get_michelson_triplet(orbits, freqs, sky_basis, paths_triplet):
    separations = get_separations(orbits)
    arms = flatten_pairs(separations)
    arm_lengths = get_arm_lengths(arms)

    response, _ = response_pipe(
        orbits,
        freqs,
        sky_basis=sky_basis,
    )
    path_responses, cumul_path_separations = get_path_response(
        paths_triplet,
        freqs,
        arm_lengths,
        response,
    )

    michelson = get_pairwise_differential_strain_response(
        path_response=path_responses, cumul_path_separations=cumul_path_separations
    )

    return michelson
