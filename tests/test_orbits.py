import jax.numpy as jnp

from jax_gw.detector.orbits import (
    create_cartwheel_orbit,
    create_cartwheel_arm_lengths,
    create_circular_orbit_xy,
    get_vertex_angle,
    earthbound_ifo_pipeline,
)
import pytest


@pytest.mark.parametrize(
    "N, len_times, test_output_shape",
    [
        (1, 1, (1, 3, 1)),
        (5, 1, (5, 3, 1)),
        (1, 5, (1, 3, 5)),
        (5, 5, (5, 3, 5)),
        (5, 10, (5, 3, 10)),
    ],
)
def test_create_cartwheel_orbit_output_shape(N, len_times, test_output_shape):
    times = jnp.linspace(0, 1, len_times)
    orbit = create_cartwheel_orbit(ecc=0, r=1, N=N, times=times)
    assert orbit.shape == test_output_shape


def test_get_vertex_angle_ifo():
    detector_lat = 0
    detector_lon = 0
    times = jnp.linspace(0, 1, 100)
    r = 1.0
    L_arm = 4.0
    psi = 0.0
    beta_arm = jnp.pi / 2
    orbits = earthbound_ifo_pipeline(
        lat=detector_lat,
        lon=detector_lon,
        times=times,
        r=r,
        L_arm=L_arm,
        psi=psi,
        beta_arm=beta_arm,
    )
    vertex_angle = get_vertex_angle(orbits) * 180 / jnp.pi
    print(vertex_angle)
    assert jnp.allclose(vertex_angle, 90, atol=1e-4)


def test_create_circular_orbit_xy_center_loc():
    times = jnp.linspace(0, 1, 10000)
    r = 3.4
    orbit = create_circular_orbit_xy(r=r, f_orb=1, times=times)
    center_x = jnp.mean(orbit, axis=1)
    assert jnp.allclose(center_x, 0, atol=1e-3)


def test_earthbound_ifo_pipeline_arm_lengths():
    detector_lat = 0
    detector_lon = 0
    times = jnp.linspace(0, 1, 100)
    r = 1.0
    L_arm = 4.0
    psi = 0.0
    beta_arm = jnp.pi / 2
    orbits = earthbound_ifo_pipeline(
        lat=detector_lat,
        lon=detector_lon,
        times=times,
        r=r,
        L_arm=L_arm,
        psi=psi,
        beta_arm=beta_arm,
    )
    km_in_AU = 149.597871 * 1e6
    distances_01 = jnp.linalg.norm(orbits[0] - orbits[1], axis=0)
    distances_01 = distances_01 * km_in_AU
    distances_02 = jnp.linalg.norm(orbits[0] - orbits[2], axis=0)
    distances_02 = distances_02 * km_in_AU
    assert jnp.allclose(distances_01, L_arm, rtol=1e-3)
    assert jnp.allclose(distances_02, L_arm, rtol=1e-3)


def test_earthbound_ifo_pipeline_orbit_AU():
    detector_lat = 0
    detector_lon = 0
    times = jnp.linspace(0, 1, 100)
    r = 1.0
    L_arm = 4.0
    psi = 0.0
    beta_arm = jnp.pi / 2
    orbits_in_AU = earthbound_ifo_pipeline(
        lat=detector_lat,
        lon=detector_lon,
        times=times,
        r=r,
        L_arm=L_arm,
        psi=psi,
        beta_arm=beta_arm,
    )
    distances_center = jnp.linalg.norm(orbits_in_AU, axis=1)
    assert jnp.allclose(distances_center, r, rtol=1e-3)


def test_earthbound_ifo_pipeline_planet_radius():
    detector_lat = 0
    detector_lon = 0
    times = jnp.linspace(0, 1, 100)
    r = 1.0
    L_arm = 4.0
    psi = 0.0
    beta_arm = jnp.pi / 2
    orbits = earthbound_ifo_pipeline(
        lat=detector_lat,
        lon=detector_lon,
        times=times,
        r=r,
        L_arm=L_arm,
        psi=psi,
        beta_arm=beta_arm,
    )
    initial_orbits = orbits[:, :, 0]
    planet_center = jnp.array([1, 0, 0])
    delta_r = initial_orbits - planet_center
    distances_center = jnp.linalg.norm(delta_r, axis=1)
    r_Earth_km = 6371.0
    km_in_AU = 149.597871 * 1e6
    assert jnp.allclose(distances_center, r_Earth_km / km_in_AU, rtol=1e-3)


def test_HL_pipeline_distance():
    H_lat, H_lon = (
        46.455140209119214 * jnp.pi / 180,
        -119.40746331631823 * jnp.pi / 180,
    )
    L_lat, L_lon = (
        30.56289433 * jnp.pi / 180,
        -90.7742404 * jnp.pi / 180,
    )
    times = jnp.linspace(0, 1, 100)
    r = 1.0
    L_arm = 4.0
    psi_H = (90 + 36) * jnp.pi / 180
    psi_L = (180 + 18) * jnp.pi / 180
    beta_arm = jnp.pi / 2
    orbits_H = earthbound_ifo_pipeline(
        lat=H_lat,
        lon=H_lon,
        times=times,
        r=r,
        L_arm=L_arm,
        psi=psi_H,
        beta_arm=beta_arm,
    )
    orbits_L = earthbound_ifo_pipeline(
        lat=L_lat,
        lon=L_lon,
        times=times,
        r=r,
        L_arm=L_arm,
        psi=psi_L,
        beta_arm=beta_arm,
    )
    distances = jnp.linalg.norm(orbits_H - orbits_L, axis=1)
    km_in_AU = 149.597871 * 1e6
    assert jnp.allclose(distances * km_in_AU, 3000, rtol=2e-3)


def test_create_cartwheel_orbit_center_loc():
    times = jnp.linspace(0, 1, 10)
    r = 3.4
    orbit = create_cartwheel_orbit(ecc=0, r=r, N=5, times=times)
    center = jnp.mean(orbit, axis=0)
    center_norm = jnp.linalg.norm(center, axis=0)
    print(center_norm)
    assert jnp.allclose(center_norm, r)


def test_create_cartwheel_orbit_distances_equal():
    times = jnp.linspace(0, 1, 10)
    r = 3.4
    orbit = create_cartwheel_orbit(ecc=0.01, r=r, N=3, times=times)
    distances_01 = jnp.linalg.norm(orbit[0] - orbit[1], axis=0)
    distances_12 = jnp.linalg.norm(orbit[1] - orbit[2], axis=0)
    distances_20 = jnp.linalg.norm(orbit[2] - orbit[0], axis=0)
    # assert close to 2 significant digits
    assert jnp.allclose(distances_01, distances_12, rtol=r * 0.01)
    assert jnp.allclose(distances_12, distances_20, rtol=r * 0.01)
    assert jnp.allclose(distances_20, distances_01, rtol=r * 0.01)


def test_create_cartwheel_orbit_distances_LISA():
    AU_per_billion_meters = 149.597871
    L_target = 2.5
    R_target = 1.0
    ecc = L_target / (AU_per_billion_meters * 2 * jnp.sqrt(3).item() * R_target)
    N_LISA = 3
    times = jnp.linspace(0, 1e5, 1000)
    orbits = create_cartwheel_orbit(ecc, R_target, N_LISA, times)
    distances_01 = (
        jnp.linalg.norm(orbits[0] - orbits[1], axis=0) * AU_per_billion_meters
    )
    distances_12 = (
        jnp.linalg.norm(orbits[1] - orbits[2], axis=0) * AU_per_billion_meters
    )
    distances_20 = (
        jnp.linalg.norm(orbits[2] - orbits[0], axis=0) * AU_per_billion_meters
    )
    distances = jnp.stack([distances_01, distances_12, distances_20])
    assert jnp.allclose(distances, L_target, rtol=jnp.mean(distances) * 0.01)


def test_create_cartwheel_orbit_ecc():
    # at least one year to assure we pass from both perihelion and aphelion
    # fine spacing to assure we get close to the actual eccentricity
    times = jnp.linspace(0, 2, 10000)
    r = 1.3
    ecc = 0.001
    orbit = create_cartwheel_orbit(ecc=ecc, r=r, N=3, times=times)
    orbit_0 = orbit[0]
    orbit_L = jnp.linalg.norm(orbit_0, axis=0)
    orbit_L_min = jnp.min(orbit_L)
    orbit_L_max = jnp.max(orbit_L)
    ecc_measurment = (orbit_L_max - orbit_L_min) / (orbit_L_max + orbit_L_min)
    assert jnp.abs(ecc_measurment - ecc) / ecc < 0.01


def test_create_cartwheel_orbit_periodicity_earth_orbit():
    # at least one year to assure we pass from both perihelion and aphelion
    # fine spacing to assure we get close to the actual eccentricity
    times = jnp.linspace(0, 1, 10000)
    r = 1.0
    ecc = 0.001
    orbit = create_cartwheel_orbit(ecc=ecc, r=r, N=3, times=times)
    orbit_initial_time = orbit[:, :, 0]
    orbit_final_time = orbit[:, :, -1]
    assert jnp.allclose(orbit_initial_time, orbit_final_time, rtol=1e-3)


def test_create_cartwheel_orbit_earth_periodicity_random_orbit():
    """Check that the orbit has periodicity 1/f"""
    # f/f_e = T_e/T = (r_e/r)**(3/2)
    r = 5.0
    f = 5.0
    times = jnp.linspace(0, 1 / f, 100)
    ecc = 0
    orbit = create_cartwheel_orbit(ecc=ecc, r=r, N=3, times=times, freq=f)
    orbit_initial_time = orbit[:, :, 0]
    orbit_final_time = orbit[:, :, -1]
    assert jnp.allclose(orbit_initial_time, orbit_final_time, rtol=1e-3)


def test_create_cartwheel_orbit_zero_ecc():
    """Check that the orbits are all the same when eccentricity is zero"""
    times = jnp.linspace(0, 1, 100)
    r = 2.0
    ecc = 0
    orbit = create_cartwheel_orbit(ecc=ecc, r=r, N=3, times=times)
    assert jnp.allclose(orbit[0], orbit[1], rtol=1e-3)


def test_create_cartwheel_orbit_nonzero_ecc():
    """Check that the orbits are not all the same when eccentricity is nonzero"""
    times = jnp.linspace(0, 1, 100)
    r = 2.0
    ecc = 0.3
    orbit = create_cartwheel_orbit(ecc=ecc, r=r, N=3, times=times)
    assert not jnp.allclose(orbit[0], orbit[1], rtol=1e-3)


def test_create_cartwheel_orbit_arm_lengths():
    """Check that analytic distances agree with numeric"""
    times = jnp.linspace(0, 1, 10)
    ecc = 0.01
    r = 1.0
    N = 3
    orbit = create_cartwheel_orbit(ecc=ecc, r=r, N=N, times=times)
    distances_01 = jnp.linalg.norm(orbit[0] - orbit[1], axis=0)
    distances_12 = jnp.linalg.norm(orbit[1] - orbit[2], axis=0)
    distances_20 = jnp.linalg.norm(orbit[2] - orbit[0], axis=0)
    # assert close to 2 significant digits
    orbit_analytic = create_cartwheel_arm_lengths(ecc=ecc, r=r, N=N, times=times)
    assert jnp.allclose(
        distances_01, orbit_analytic[:, 0, 1], rtol=1e-2 * distances_01[0]
    )
    assert jnp.allclose(
        distances_12, orbit_analytic[:, 1, 2], rtol=1e-2 * distances_12[0]
    )
    assert jnp.allclose(
        distances_20, orbit_analytic[:, 2, 0], rtol=1e-2 * distances_20[0]
    )
