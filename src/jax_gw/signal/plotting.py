from jax import numpy as jnp

from jax_gw.detector.pixel import unflatten_sky_axis


def plot_response(
    plotted_response, ecl_thetas_reduced, ecl_phis_reduced, ax, unflatten=True
):
    """Plot the response function for a given source direction.

    Parameters
    ----------
    plotted_response : jnp.array
        Response function for a given source direction.
    ecl_thetas_reduced : jnp.array
        Reduced ecliptic latitudes.
    ecl_phis_reduced : jnp.array
        Reduced ecliptic longitudes.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    """
    N_theta, N_phi = ecl_thetas_reduced.size, ecl_phis_reduced.size
    if unflatten:
        plotted_response = unflatten_sky_axis(
            plotted_response, axis=0, N_theta=N_theta, N_phi=N_phi
        )
    u = ecl_phis_reduced
    v = ecl_thetas_reduced
    x = plotted_response * jnp.outer(jnp.sin(v), jnp.cos(u))
    y = plotted_response * jnp.outer(jnp.sin(v), jnp.sin(u))
    z = plotted_response * jnp.outer(jnp.cos(v), jnp.ones(jnp.size(u)))

    # Plot the surface
    ax.plot_surface(x, y, z, zorder=0, alpha=0.3, color="C0")

    # Set an equal aspect ratio
    ax.set_aspect("equal")
