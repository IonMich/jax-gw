from jax import Array
from jax.typing import ArrayLike

import jax.numpy as jnp

from jax.scipy.integrate import trapezoid


def unpolarized_cross_overlap(michelson_1: ArrayLike, michelson_2: ArrayLike) -> Array:
    """Overlap between two Michelson responses.

    Parameters
    ----------
    michelson_1: ArrayLike
        First Michelson response. Shape (..., N_pol)
    michelson_2: ArrayLike
        Second Michelson response. Shape (..., N_pol)

    Returns
    -------
    overlap: Array
        Overlap between the two Michelson responses.
    """

    return 0.5 * jnp.sum(
        michelson_1 * jnp.conj(michelson_2),
        axis=-1,
    )


def unpolarized_auto_overlap(michelson: ArrayLike) -> Array:
    """Overlap between two Michelson responses.

    Parameters
    ----------
    michelson: ArrayLike
        Michelson response. Shape (..., N_pol)

    Returns
    -------
    overlap: Array
        Overlap between the two Michelson responses.
    """

    return unpolarized_cross_overlap(michelson, michelson)


def overlap_angular_noise_ell(
    overlap_lm, noise_psd_f, ell_array, freqs, f_ref, spectral_index, t_obs
):
    spectral_shape = jnp.power(freqs / f_ref, spectral_index)
    # TODO: using just t=0 for now, and approximating
    # the temporal integration. This should be fixed.
    overlap_init = overlap_lm[0]
    # sum the square complex norm over m (axis=-1)
    overlap_sq_init = jnp.sum(overlap_init * jnp.conj(overlap_init), axis=-1)
    integrand = (
        overlap_sq_init
        * ((2.0 / 5.0 * spectral_shape / noise_psd_f) ** 2)[:, None]
        / (2 * ell_array + 1)
    )
    # integrate over frequency
    noise_ell_inv = t_obs / 2 * trapezoid(integrand, freqs, axis=0)

    return noise_ell_inv ** (-1)
