import jax.numpy as jnp


def WD_binary_source(
    A: float, f_0: float, f_0_dot: float, phi_0: float, times: jnp.array
) -> jnp.array:
    """Create the plus and cross polarizations for a WD binary at the source frame.

    Parameters
    ----------
    A : float
        Amplitude of the wave.
    f_0 : float
        Frequency of the wave.
    f_0_dot : float
        Frequency derivative of the wave.
    phi_0 : float
        Initial phase of the wave.
    times : jnp.array
        Times at which to evaluate the orbit.

    Returns
    -------
    jnp.array
        Plus and cross polarization waveforms. Dimensions: (2, len(times)).
    """
    years_to_seconds = 365.25 * 24.0 * 3600.0
    f_double_dot = 11.0 / 3.0 * f_0_dot**2 / f_0
    times = times * years_to_seconds

    phase_1 = f_0 * times
    phase_2 = 0.5 * f_double_dot * times**2
    phase_3 = 1.0 / 6.0 * f_0_dot * times**3

    phase = 2.0 * jnp.pi * (phase_1 + phase_2 + phase_3) + phi_0

    plus = A * jnp.cos(phase)
    cross = A * jnp.sin(phase)

    return jnp.stack([plus, cross], axis=1)


def source_to_detector_frame(
    iota: float, psi: float, waveforms: jnp.array
) -> jnp.array:
    """Convert the plus and cross polarizations from the source frame to the detector frame.

    Parameters
    ----------
    iota : float
        Inclination of the source.
    psi : float
        Polarization angle of the source.
    waveforms : jnp.array
        Plus and cross polarization waveforms. Dimensions: (2, len(times)).

    Returns
    -------
    jnp.array
        Plus and cross polarization waveforms. Dimensions: (2, len(times)).
    """
    cos_iota = jnp.cos(iota)

    plus_proj = -(1.0 + cos_iota**2) * waveforms[:, 0]
    cross_proj = -2.0 * cos_iota * waveforms[:, 1]

    plus = jnp.cos(2.0 * psi) * plus_proj - jnp.sin(2.0 * psi) * cross_proj
    cross = jnp.sin(2.0 * psi) * plus_proj + jnp.cos(2.0 * psi) * cross_proj

    return jnp.stack([plus, cross], axis=1)
