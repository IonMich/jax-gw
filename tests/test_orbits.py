import numpy as np

from jax_gw.detector.orbits import create_cartwheel_orbit
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
    times = np.linspace(0, 1, len_times)
    orbit = create_cartwheel_orbit(ecc=0, r=1, N=N, times=times)
    assert orbit.shape == test_output_shape
