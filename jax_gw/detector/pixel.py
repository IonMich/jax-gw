"""Pixelization scheme module for jax-gw.

This module contains functions for calculating the sky geometry.
"""
import jax.numpy as jnp


def get_directional_basis(ecl_theta: float, ecl_phi: float) -> jnp.array:
    """Calculate the directional basis for a given source direction.

    Parameters
    ----------
    ecl_theta : float
        Ecliptic latitude of the source.
    ecl_phi : float
        Ecliptic phi of the source.

    Returns
    -------
    jnp.array
        Directional basis k_hat, u_hat, v_hat, where k_hat is the direction of the
        incoming signal, u_hat is same as theta_hat, and v_hat is same as phi_hat.
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


def unflatten_sky_axis(matrix, axis: int, N_theta: int, N_phi: int) -> jnp.array:
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
