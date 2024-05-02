"""Pixelization scheme module for jax-gw.

This module contains functions for calculating the sky geometry.
"""

import jax
from jax import Array
import jax.numpy as jnp
from jax.scipy.special import lpmn_values
from jax.typing import ArrayLike


def get_directional_basis(ecl_theta: ArrayLike, ecl_phi: ArrayLike) -> Array:
    """Calculate the directional basis for a given source direction.

    Parameters
    ----------
    ecl_theta : ArrayLike
        Ecliptic latitude of the source.
    ecl_phi : ArrayLike
        Ecliptic phi of the source.

    Returns
    -------
    jnp.array
        Directional basis k_hat, u_hat, v_hat, where k_hat is the direction of the
        incoming signal, u_hat is same as theta_hat, and v_hat is same as phi_hat.

        Note that k, u, v are not a right-handed coordinate system, but -k, u, v is.
    """
    cos_theta = jnp.cos(ecl_theta)
    sin_theta = jnp.sin(ecl_theta)
    cos_phi = jnp.cos(ecl_phi)
    sin_phi = jnp.sin(ecl_phi)
    zero_element = jnp.zeros_like(cos_theta)

    k_hat = -jnp.stack(
        [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta],
        axis=-1,
    )
    # u_hat is theta_hat, v_hat is phi_hat
    u_hat = jnp.stack([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], axis=-1)
    v_hat = jnp.stack([-sin_phi, cos_phi, zero_element], axis=-1)

    return jnp.stack([k_hat, u_hat, v_hat], axis=0)


def flatten_sky(i_theta: int, j_phi: int, N_phi: int) -> int:
    """Flatten the sky coordinates into a single index.

    Parameters
    ----------
    i_theta : int
        Index of the ecliptic theta.
    j_phi : int
        Index of the ecliptic phi.
    N_phi : int
        Number of ecliptic phis.

    Returns
    -------
    int
        Flattened index.
    """
    return i_theta * N_phi + j_phi


def unflatten_sky(index: int, N_phi: int):
    """Unflatten the sky coordinates from a single index.

    Parameters
    ----------
    index : int
        Flattened index.
    N_phi : int
        Number of ecliptic phis.

    Returns
    -------
    NamedTuple
        Unflattened sky coordinates.
    """
    i_theta = index // N_phi
    j_phi = index % N_phi

    return i_theta, j_phi


def flat_to_matrix_sky_indices(N_theta: int, N_phi: int):
    """Calculate the (N_theta*N_phi, 2) matrix of flat indices for a given sky resolution.

    Parameters
    ----------
    N_theta : int
        Number of ecliptic thetas.
    N_phi : int
        Number of ecliptic phis.

    Returns
    -------
    ArrayLike
        Matrix of flat indices.
    """
    # without for loop or list comprehension
    a = jnp.arange(N_theta * N_phi)

    i = jnp.floor_divide(a, N_phi)
    j = jnp.mod(a, N_phi)

    return jnp.stack([i, j], axis=1)


def unflatten_sky_axis(matrix, axis: int, N_theta: int, N_phi: int) -> Array:
    """Unflatten the axis of a matrix that corresponds to the sky coordinates.

    Shape is converted from (...N, N_theta*N_phi, M...) to (...N, N_theta, N_phi, M...).

    Parameters
    ----------
    matrix : ArrayLike
        Matrix to unflatten.
    axis : int
        Axis to unflatten.
    N_theta : int
        Number of ecliptic thetas.
    N_phi : int
        Number of ecliptic phis.

    Returns
    -------
    jnp.array
        Unflattened matrix.
    """
    flat_to_matrix = jnp.arange(N_theta * N_phi).reshape(N_theta, N_phi)

    return jnp.take(matrix, flat_to_matrix, axis=axis)


def get_sph_harm_values(l_max, ecl_thetas_reduced, ecl_phis_reduced):
    lpmn_l_max_jitted = jax.jit(
        lambda x: lpmn_values(l_max, l_max, x, is_normalized=True)
    )
    alp_normed = lpmn_l_max_jitted(jnp.cos(ecl_thetas_reduced))
    # swap the first two axes to have the l axis first
    alp_normed = jnp.swapaxes(alp_normed, 0, 1)
    exp_1j_m_phi = jnp.exp(1j * jnp.outer(jnp.arange(0, l_max + 1), ecl_phis_reduced))
    sph_harm_values = alp_normed[..., None] * exp_1j_m_phi[None, :, None, :]
    sph_harm_values = sph_harm_values.reshape(*sph_harm_values.shape[:-2], -1)
    return sph_harm_values


def get_solid_angle_theta_phi(theta, phi, N_theta, N_phi):
    """Get the sky area associated with a given theta and phi in a
    pixelated sphere. Assumes linear spacing in theta and phi.

    Parameters
    ----------
    theta : float
        Ecliptic theta.
    phi : float
        Ecliptic phi.
    N_theta : int
        Number of theta bins.
    N_phi : int
        Number of phi bins.

    Returns
    -------
    float
        Sky area associated with theta and phi.
    """
    delta_phi = 2 * jnp.pi / N_phi
    delta_theta = jnp.pi / (N_theta - 1)
    min_phi = phi - delta_phi / 2
    max_phi = phi + delta_phi / 2
    min_theta = jnp.maximum(theta - delta_theta / 2, 0)
    max_theta = jnp.minimum(theta + delta_theta / 2, jnp.pi)
    solid_angle = (max_phi - min_phi) * (jnp.cos(min_theta) - jnp.cos(max_theta))
    return solid_angle


def pixel_to_lm(
    data_omega, axis, N_theta, N_phi, ecl_thetas, ecl_phis, sph_harm_values
):
    """Convert a pixelated map to a spherical harmonic map."""
    # sky axis last, preceded by two axes for l and m
    data_omega = jnp.moveaxis(data_omega, axis, -1)[..., None, None, :]
    data_lm = sph_harm_values * data_omega
    data_lm = data_lm * get_solid_angle_theta_phi(ecl_thetas, ecl_phis, N_theta, N_phi)
    data_lm = jnp.sum(data_lm, axis=-1)
    return data_lm
