import jax
from jax import Array
from jax.typing import ArrayLike
from jax import config

import jax.numpy as jnp

config.update("jax_enable_x64", True)

# in 1/yr
FREQ_EARTH_ORBIT = 1.0

EARTH_TILT = 23.44 * jnp.pi / 180.0
# lat and lon on the Earth needed
# for object with initial x=0, y=0 in ecliptic coordinates
EARTH_Z_LAT = jnp.pi / 2.0 - EARTH_TILT
EARTH_Z_LON = -jnp.pi / 2.0


def create_circular_orbit_xy(r: float, f_orb: float, times: ArrayLike) -> Array:
    """Create an orbit around the Sun with x and y Arms.

    Parameters
    ----------
    r : float
        Radius of the orbit.
    times : jnp.array
        Times at which to evaluate the orbit.

    Returns
    -------
    jnp.array
        Orbit. Dimensions: (1, 3, len(times)).
    """
    # for now let's assume a circular orbit on the ecliptic plane
    x = r * jnp.cos(2.0 * jnp.pi * f_orb * times)
    y = r * jnp.sin(2.0 * jnp.pi * f_orb * times)
    z = jnp.zeros_like(x)

    return jnp.stack([x, y, z], axis=0)


def lat_lon_to_cartesian(lat: float, lon: float, r: float = 1) -> Array:
    """Convert latitude and longitude to equatorial cartesian coordinates.

    Parameters
    ----------
    lat : float
        Latitude in radians.
    lon : float
        Longitude in radians.
    r : float
        Radius.

    Returns
    -------
    jnp.array
        Equatorial cartesian coordinates.
    """
    x = r * jnp.cos(lat) * jnp.cos(lon)
    y = r * jnp.cos(lat) * jnp.sin(lon)
    z = r * jnp.sin(lat)

    return jnp.stack([x, y, z], axis=0)


def equatorial_timeshift(equatorial_coords: ArrayLike, angle: ArrayLike) -> Array:
    """Rotate a vector by an angle `angle` around the z-axis of equatorial coordinates.
    Shift equatorial coordinates to a hour `angle` later.

    Parameters
    ----------
    equatorial_coords : ArrayLike
        Vector in equatorial coordinates.
    angle : ArrayLike
        Angle to rotate around the z-axis.

    Returns
    -------
    Array
        Vector in equatorial coordinates at time shifted by `angle`.
    """
    x, y, z = equatorial_coords
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    x_return = cos_angle * x + sin_angle * y
    y_return = -sin_angle * x + cos_angle * y
    z_return = z * jnp.ones_like(x_return)
    return jnp.stack([x_return, y_return, z_return], axis=0)


def axial_tilt(equatorial_coords: ArrayLike, earth_tilt: float) -> Array:
    """Rotate a vector by an angle `tilt` around the x-axis.
    Convert from equatorial to ecliptic coordinates when `earth_tilt` is the positive Earth's tilt.

    Parameters
    ----------
    equatorial_coords : ArrayLike
        Vector in equatorial coordinates.
    earth_tilt : float
        Angle to rotate around the x-axis.

    Returns
    -------
    Array
        Vector in ecliptic coordinates.
    """
    rot_matrix = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, jnp.cos(earth_tilt), jnp.sin(earth_tilt)],
            [0.0, -jnp.sin(earth_tilt), jnp.cos(earth_tilt)],
        ]
    )
    return jnp.dot(rot_matrix, equatorial_coords)


def ecliptic_timeshift(
    ecliptic_coords: ArrayLike, angle: ArrayLike, tilt: float
) -> Array:
    """Rotate a vector in ecliptic coordinates by an angle `angle` around the z-axis of equatorial coordinates.
    Shift ecliptic coordinates to a hour `angle` later.

    Parameters
    ----------
    ecliptic_coords : ArrayLike
        Vector in ecliptic  coordinates.
    angle : ArrayLike
        Angle to rotate around the z-axis of equatorial coordinates.
    tilt : float
        Angle to rotate around the x-axis.

    Returns
    -------
    Array
        Vector in equatorial coordinates at time shifted by `angle`.
    """
    equatorial_initial = axial_tilt(ecliptic_coords, -tilt)

    equatorial_coords = equatorial_timeshift(equatorial_initial, angle)
    ecliptic_coords = axial_tilt(equatorial_coords, tilt)

    return ecliptic_coords


def create_cartwheel_orbit(
    ecc: float,
    r: float,
    N: int,
    times: ArrayLike,
    timeshift: float = 0,
    freq: float = 1.0,
) -> Array:
    """Create a cartwheel orbit.

    Parameters
    ----------
    ecc : float
        Eccentricity of the orbits.
    r : float
        Radius of the orbit of the guiding center. Units: AU.
    N : int
        Number of spacecraft.
    times : ArrayLike
        Times at which to evaluate the orbit. Units: years.

    Returns
    -------
    Array
        Orbit. Dimensions: (N, 3, len(times)). Units: AU.
    """
    # kappa is 20 degrees behind Earth
    kappa_orbit = -20.0 / 360.0 * 2 * jnp.pi
    lambda_cart = timeshift
    alpha = 2.0 * jnp.pi * freq * times + kappa_orbit
    beta_n = jnp.arange(N)[:, jnp.newaxis] * 2.0 * jnp.pi / N + lambda_cart

    exp_1_0 = jnp.exp(1j * (alpha))
    exp_2_n1 = jnp.exp(1j * (2 * alpha - beta_n))
    exp_0_1 = jnp.exp(1j * (beta_n))
    exp_3_n2 = jnp.exp(1j * (3 * alpha - 2 * beta_n))
    exp_1_n2 = jnp.exp(1j * (alpha - 2 * beta_n))

    term_1 = r * exp_1_0
    term_2 = 0.5 * r * ecc * (exp_2_n1 - 3.0 * exp_0_1)
    term_3 = 0.125 * r * ecc**2 * (3.0 * exp_3_n2 - 10.0 * exp_1_0)
    term_4 = 0.125 * r * ecc**2 * 5.0 * exp_1_n2

    L = jnp.sqrt(3) * ecc * r
    exp_1_n1 = jnp.exp(1j * (alpha - beta_n))
    cos_1_n1, sin_1_n1 = jnp.real(exp_1_n1), jnp.imag(exp_1_n1)

    common_x_y = term_1 + term_2 + term_3

    x = jnp.real(common_x_y - term_4)
    y = jnp.imag(common_x_y + term_4)
    z = -L * cos_1_n1 + L * ecc * (1 + sin_1_n1**2)

    return jnp.stack([x, y, z], axis=1)


def create_cartwheel_arm_lengths(
    ecc: float,
    r: float,
    N: int,
    times: ArrayLike,
    freq: float = 1.0,
) -> Array:
    """Create the scalar separations for a cartwheel orbit.

    Parameters
    ----------
    ecc : float
        Eccentricity of the orbit.
    r : float
        Radius of the orbit of the guiding center.
    N : int
        Number of spacecraft.
    times : jnp.array
        Times at which to evaluate the orbit.

    Returns
    -------
    jnp.array
        Separations. Dimensions: (len(times), N, N).
    """
    assert N == 3
    L = 2.0 * jnp.sqrt(3) * ecc * r

    lambda_cart = 0.0
    kappa_orbit = -20.0 / 360.0 * 2 * jnp.pi
    alpha = 2.0 * jnp.pi * freq * times + kappa_orbit

    exp_1_n1 = jnp.exp(1j * (alpha - lambda_cart))
    cos_1_n1 = jnp.real(exp_1_n1)
    cos_3_n3 = jnp.real(exp_1_n1**3)
    sin_1_n1_pi6 = jnp.imag(exp_1_n1 * jnp.exp(1j * jnp.pi / 6.0))
    sin_1_n1_npi6 = jnp.imag(exp_1_n1 * jnp.exp(-1j * jnp.pi / 6.0))

    arm_12 = L * (1 + ecc / 32.0 * (15.0 * sin_1_n1_pi6 - cos_3_n3))
    arm_13 = L * (1 - ecc / 32.0 * (15.0 * sin_1_n1_npi6 + cos_3_n3))
    arm_23 = L * (1 - ecc / 32.0 * (15.0 * cos_1_n1 + cos_3_n3))

    separations_flat = jnp.stack([arm_12, arm_13, arm_23], axis=0)

    d = jnp.zeros((N, N, len(times)))
    d = d.at[0, 1, :].set(separations_flat[0, :])
    d = d.at[1, 0, :].set(separations_flat[0, :])
    d = d.at[0, 2, :].set(separations_flat[1, :])
    d = d.at[2, 0, :].set(separations_flat[1, :])
    d = d.at[1, 2, :].set(separations_flat[2, :])
    d = d.at[2, 1, :].set(separations_flat[2, :])

    # move the time axis to the front
    d = jnp.moveaxis(d, -1, 0)

    return d


def get_separations(orbits: ArrayLike) -> Array:
    """Calculate the vector separations between the spacecraft.

    `r_{ij} = r_i - r_j`

    Parameters
    ----------
    orbits : ArrayLike
        Array of shape `(N, 3, N_steps)` containing the orbits of the N
        spacecraft.

    Returns
    -------
    Array
        Vector separations. Dimensions: `(N_steps, N, N, 3)`.
    """
    # calculate the vector separations
    N_steps, N = orbits.shape[2], orbits.shape[0]
    r = jnp.zeros((N_steps, N, N, 3))
    for i in range(N):
        for j in range(N):
            r = r.at[:, i, j, :].set(jnp.transpose(orbits[i, :, :] - orbits[j, :, :]))

    return r


def get_arm_lengths(separations: ArrayLike) -> Array:
    """Calculate the arm lengths from the vector separations.

    Parameters
    ----------
    separations : jnp.ndarray
        Vector separations. Last dimension must be of length 3.

    Returns
    -------
    jnp.ndarray
        Arm lengths in shape `(N_steps, N, N)`.
    """
    d = jnp.linalg.norm(separations, axis=-1)

    return d


def get_receiver_positions(
    position: ArrayLike,
) -> Array:
    """Calculate the receiver positions of the spacecraft for a collection of arms.
    Since the separation matrix is defined as `r[i, j] = r[i] - r[j]`, the
    receiver positions must be defined via `r_pos[i, j, 3, N...] = pos[i, 3, N...]`.
    Positions has shape `(N, 3, N_steps)` while separations has shape
    `(N_steps, N, N, 3)`. Therefore, we need to create a newaxis for the `j` index,
    and finally move the time axis to the front.

    Parameters
    ----------
    position : ArrayLike
        Position of the spacecraft. Shape (N, 3, ...).

    Returns
    -------
    Array
        Receiver spacecraft positions for the arm. Shape broadcastable to
        the shape of separations.
    """

    # first create a newaxis for the j index
    pos = position[:, jnp.newaxis, :, ...]
    # then move the time axis to the front, if it exists
    from_idx = [0, 1, 2]
    to_idx = [-3, -2, -1]
    pos = jnp.moveaxis(pos, from_idx, to_idx)

    return pos


def get_emitter_positions(
    position: ArrayLike,
) -> Array:
    """Calculate the emitter positions of the spacecraft for a given arm.
    Since the separation matrix is defined as r[i, j] = r[i] - r[j], the
    emitter positions must be calculated as e_pos[i, j, 3, N...] = pos[j, 3, N...].
    Parameters
    ----------
    position : jnp.array
        Position of the spacecraft. Shape (N, 3, ...).

    Returns
    -------
    jnp.array
        Emitter spacecraft positions for the arm. Same shape as separations.
    """
    # first create a newaxis for the i index
    pos = position[jnp.newaxis, :, :, ...]
    # then move the time axis to the front, if it exists
    from_idx = [0, 1, 2]
    to_idx = [-3, -2, -1]
    pos = jnp.moveaxis(pos, from_idx, to_idx)

    return pos


def flat_index(i: jnp.int32, j: jnp.int32, N: int) -> jnp.int32:
    """Calculate the flat index for a pair of indices i, j.

    Parameters
    ----------
    i : jnp.int32
        First index.
    j : jnp.int32
        Second index.
    N : int
        Possible values for i (or j).

    Returns
    -------
    jnp.int32
        Flat index for the pair of indices corresponding to the pair of
        spacecraft.
    """
    min_ij = jnp.minimum(i, j)
    max_ij = jnp.maximum(i, j)
    # returns 0 if i < j and 1 if i > j
    really_fast_index = jnp.greater(i, j)
    fast_index = 2 * (max_ij - min_ij - 1)
    slow_index = min_ij * (2 * N - min_ij - 1)

    return really_fast_index + fast_index + slow_index


def flat_to_matrix_indices(
    N: int,
) -> Array:
    """Calculate the (N*(N-1), 2) matrix of flat indices for a given number of
    spacecraft.

    Parameters
    ----------
    N : int
        Number of spacecraft.

    Returns
    -------
    jnp.array
        Matrix of flat indices.
    """
    # create the matrix of flat indices
    flat_indices = jnp.zeros((N * (N - 1), 2), dtype=jnp.int32)

    index_func = jax.jit(flat_index)
    for i, j in zip(*jnp.triu_indices(N, k=1)):
        k = index_func(i, j, N)
        flat_indices = flat_indices.at[k, :].set(jnp.stack([i, j], axis=0))
        k = index_func(j, i, N)
        flat_indices = flat_indices.at[k, :].set(jnp.stack([j, i], axis=0))

    return flat_indices


@jax.jit
def flatten_pairs(
    matrix_form: ArrayLike,
) -> Array:
    """Flatten the separations or receiver positions from a pair of indices
    to a single dimension of length N * (N - 1).

    Parameters
    ----------
    matrix_form : jnp.array
        Separations or receiver positions in matrix form.

    Returns
    -------
    jnp.array
        Flattened separations and receiver positions with shape (N_steps, N * (N - 1), 3).
    """
    N1, N2 = matrix_form.shape[1], matrix_form.shape[2]
    N = max(N1, N2)
    # use of minimum is to avoid out of bounds error when N1 or N2 has length 1
    vmapped_flat_index = jax.vmap(
        lambda i, j: matrix_form[:, jnp.minimum(i, N1), jnp.minimum(j, N2), ...],
        in_axes=(0, 0),
        out_axes=0,
    )
    indices = flat_to_matrix_indices(N)
    receivers, emitters = indices[:, 0], indices[:, 1]
    return vmapped_flat_index(receivers, emitters)


def path_from_indices(indices: ArrayLike) -> Array:
    """Convert an array of indices of spacecraft and length N_depth+1 to a path
    that is a 1D array of length N_depth and contains the arm index for each part of the path.

    Parameters
    ----------
    indices : jnp.array
        Array of indices of spacecraft and length N_depth+1.

    Returns
    -------
    jnp.array
        Path that is a 1D array of length N_depth and contains the arm index for each part of the path.
    """
    # first from (N_paths, N_depth+1,) to (N_paths, N_depth+1, 2), where the last axis contains the indices
    # shifted by one, so that (..., i, 0) is the start and (..., i, 1) is the end of the segment i of the path
    indices = jnp.stack([indices, jnp.roll(indices, -1, axis=-1)], axis=-1)
    # remove the last row
    indices = indices[..., :-1, :]

    return indices


def earthbound_ifo_pipeline(
    lat: float,
    lon: float,
    times: ArrayLike,
    r: float,
    L_arm: float,
    psi: float = 0,
    beta_arm: float = jnp.pi / 2,
) -> Array:
    """Create the orbits of the spacecraft for an Earthbound interferometer.
    Currently only works for perpendicular arms and assumes a circular orbit
    with the Earth modeled as a sphere.

    Parameters
    ----------
    lat : float
        Earth latitude in radians. Zero is the equator, +pi/2 is the North pole.
    lon : float
        Earth longitude in radians. Zero is the Greenwich meridian, positive is East.
    times : jnp.array
        Times at which to evaluate the orbit in years.
    r : float
        Radius of the orbit in AU.
    L_arm : float
        Length of the arms in km.
    psi : float
        Angle between the X arm and local East in radians. Positive North of East. The Y arm is rotated by an additional pi/2.
    beta_arm : float
        Angle between the X and Y arms in radians.

    Returns
    -------
    jnp.array
        Orbits of the N=3 points defining the interferometer. Dimensions: (N, 3, len(times)).
    """
    FREQ_CENTER_ORBIT = 1  # in 1/year
    FREQ_ROTATION = 365.25  # in 1/year
    r_orbital = create_circular_orbit_xy(r, FREQ_CENTER_ORBIT, times)

    # calculate x, y, z coordinates of detector with respect to the guiding center
    # at time t=0
    r_detector_initial_equatorial = lat_lon_to_cartesian(lat, lon)

    hour_angle = 2.0 * jnp.pi * FREQ_ROTATION * times
    r_detector = equatorial_timeshift(r_detector_initial_equatorial, hour_angle)

    r_detector = axial_tilt(r_detector, EARTH_TILT)

    r_earth_in_km = 6371.0
    # local East unit direction at the detector
    north_pole_equatorial = jnp.array([0.0, 0.0, 1.0])
    local_east = jnp.cross(north_pole_equatorial, r_detector_initial_equatorial)
    local_east = local_east / jnp.linalg.norm(local_east)
    # rotate the arms by psi with respect to r_detector_initial_equatorial
    # by applying the matrix form of Rodrigues' rotation formula
    K_matrix = jnp.array(
        [
            [0.0, -r_detector_initial_equatorial[2], r_detector_initial_equatorial[1]],
            [r_detector_initial_equatorial[2], 0.0, -r_detector_initial_equatorial[0]],
            [-r_detector_initial_equatorial[1], r_detector_initial_equatorial[0], 0.0],
        ]
    )
    rotation_matrix_psi = (
        jnp.eye(3) + jnp.sin(psi) * K_matrix + (1 - jnp.cos(psi)) * K_matrix @ K_matrix
    )
    rotation_matrix_beta = (
        jnp.eye(3)
        + jnp.sin(beta_arm) * K_matrix
        + (1 - jnp.cos(beta_arm)) * K_matrix @ K_matrix
    )
    x_arm_direction = rotation_matrix_psi @ local_east
    y_arm_direction = rotation_matrix_beta @ x_arm_direction
    print(x_arm_direction)
    print(y_arm_direction)
    arm_length = L_arm / r_earth_in_km
    x_arm_local_equatorial_initial = arm_length * x_arm_direction
    y_arm_local_equatorial_initial = arm_length * y_arm_direction
    # convert from equatorial to ecliptic coordinates
    x_arm_ecliptic_initial = axial_tilt(x_arm_local_equatorial_initial, +EARTH_TILT)
    print(x_arm_ecliptic_initial)
    y_arm_ecliptic_initial = axial_tilt(y_arm_local_equatorial_initial, +EARTH_TILT)
    print(y_arm_ecliptic_initial)

    # x_arm_ecliptic_initial = jnp.array([L_arm / r_earth_in_km, 0.0, 0.0])
    # y_arm_ecliptic_initial = jnp.array([0.0, L_arm / r_earth_in_km, 0.0])
    x_arm = ecliptic_timeshift(x_arm_ecliptic_initial, hour_angle, EARTH_TILT)
    y_arm = ecliptic_timeshift(y_arm_ecliptic_initial, hour_angle, EARTH_TILT)

    # add a rotation around this guiding center, assuming a solid body like the Earth
    earth_radius_per_km = 6371.0
    AU_per_billion_meters = 149.597871
    AU_per_earth_radius = (AU_per_billion_meters * 1e9) / (earth_radius_per_km * 1e3)
    print(AU_per_earth_radius)

    r_detector = jnp.array(r_detector, dtype=jnp.float64)
    r_beam_splitter = r_orbital + r_detector / AU_per_earth_radius

    x_arm = jnp.array(x_arm, dtype=jnp.float64) / AU_per_earth_radius
    y_arm = jnp.array(y_arm, dtype=jnp.float64) / AU_per_earth_radius
    x_arm = r_beam_splitter + x_arm
    y_arm = r_beam_splitter + y_arm

    orbits = jnp.stack([r_beam_splitter, x_arm, y_arm], axis=0)

    return orbits
