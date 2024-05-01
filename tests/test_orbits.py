import jax.numpy as jnp

from jax_gw.detector.orbits import create_cartwheel_orbit, create_cartwheel_arm_lengths
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
