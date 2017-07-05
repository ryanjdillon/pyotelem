
def calc_PRH(ax, ay, az):
    '''Calculate the pitch, roll and heading for triaxial movement signalsi

    Args
    ----
    ax: ndarray
        x-axis acceleration values
    ay: ndarray
        y-axis acceleration values
    az: ndarray
        z-axis acceleration values

    Returns
    -------
    pitch: ndarray
        Pitch angle in radians
    roll: ndarray
        Pitch angle in radians
    yaw: ndarray
        Pitch angle in radians
    '''

    pitch = calc_pitch(ax, ay, az)
    roll  = calc_roll(ax, ay, az)
    yaw   = calc_yaw(ax, ay, az)

    #from dtag_toolbox_python.dtag2 import a2pr
    #pitch, roll, A_norm = a2pr.a2pr(A_g)

    return pitch, roll, yaw


def calc_pitch(ax, ay, az):
    '''Angle of x-axis relative to ground (theta)

    Args
    ----
    ax: ndarray
        x-axis acceleration values
    ay: ndarray
        y-axis acceleration values
    az: ndarray
        z-axis acceleration values

    Returns
    -------
    pitch: ndarray
        Pitch angle in radians
    '''
    import numpy
    # arctan2 not needed here to cover all quadrants, just for consistency
    return numpy.arctan(ax, numpy.sqrt(ay**2+az**2))


def calc_roll(ax, ay, az):
    '''Angle of y-axis relative to ground (phi)

    Args
    ----
    ax: ndarray
        x-axis acceleration values
    ay: ndarray
        y-axis acceleration values
    az: ndarray
        z-axis acceleration values

    Returns
    -------
    roll: ndarray
        Roll angle in radians
    '''
    import numpy
    #return numpy.arctan(ay/numpy.sqrt(ax**2+az**2))
    return numpy.arctan2(ay,az)


def calc_yaw(ax, ay, az):
    '''Angle of z-axis relative to ground (psi)

    Args
    ----
    ax: ndarray
        x-axis acceleration values
    ay: ndarray
        y-axis acceleration values
    az: ndarray
        z-axis acceleration values

    Returns
    -------
    yaw: ndarray
        Yaw angle in radians
    '''
    import numpy
    #return numpy.arctan(numpy.sqrt(ax**2+ay**2)/az)
    return numpy.arctan2(ax,ay)


def absdeg(deg):
    '''Change from signed degrees to 0-180 or 0-360 ranges

    deg: ndarray
        Movement data in pitch, roll, yaw (degrees)

    Returns
    -------
    deg_abs: ndarray
        Movement translated from -180:180/-90:90 degrees to 0:360/0:180 degrees

    Example
    -------
    deg = numpy.array([-170, -120, 0, 90])
    absdeg(deg) # returns array([190, 240,   0,  90])
    '''
    import numpy

    d = numpy.copy(deg)

    if numpy.max(numpy.abs(deg)) > 90.0:
        d[deg < 0] = 360 + deg[deg < 0]
    else:
        d[deg < 0] = 180 + deg[deg < 0]

    return d


def acceleration_magnitude(ax, ay, az):
    '''Cacluate the magnitude of 3D acceleration

    Args
    ----
    ax: ndarray
        x-axis acceleration values
    ay: ndarray
        y-axis acceleration values
    az: ndarray
        z-axis acceleration values

    Returns
    -------
    acc_mag: ndarray
        Magnitude of acceleration from combined acceleration axes

    http://physics.stackexchange.com/a/41655/126878
    '''
    import numpy
    return numpy.sqrt(ax**2 + ay**2 + az**2)


def triaxial_integration(x, y, z, initial=0):
    '''Integrate tri-axial vector data

    The integration of acceleration is the velocity, and the integration of
    velocity is the position.

    x: ndarray
        X-axis component of vectors
    y: ndarray
        Y-axis component of vectors
    z: ndarray
        Z-axis component of vectors

    Returns
    -------
    x_int: ndarray
        Integration of x-axis component of vectors
    y_int: ndarray
        Integration of y-axis component of vectors
    z_int: ndarray
        Integration of z-axis component of vectors
    '''
    import scipy.integrate

    x_int = scipy.integrate.cumtrapz(range(len(x)), x, initial=initial)
    y_int = scipy.integrate.cumtrapz(range(len(y)), y, initial=initial)
    z_int = scipy.integrate.cumtrapz(range(len(z)), z, initial=initial)

    return x_int, y_int, z_int


def speed_from_acc_and_ref(acc_x, fs_a, rel_speed, zero_level,
        theoretic_max=None, rm_neg=True):
    '''Estimate speed from x-axis acceleration, fitting to relative speed data

    Args
    ----
    acc_x: ndarray
        x-axis acceleromter data, same axis of movement as relative speed sensor
    fs_a: float
        sampling frequency of accelerometer sensor
    rel_speed: ndarray
        relative speed sensor data at same sampling freqency as accelerometer
    zero_level: float
        threshold at which relative speed data should be considered zero m/s^2
    theoretic_max: float
        theoretical maximum speed of the animal

    Returns
    -------
    speed_cal: ndarray
        relative speed data calibrated to the curve fit to the integrated speed
        from the accelerometer data

    Note
    ----
    The relative speed data is fit to the curve generated from the derived
    speed from the acceleration signal.

    If a theoritc maximum speed is given, the gain of this fit is adjusted such
    that the maximum value of the resulting speed array equals the animal's
    theoretical maximum speed.This assumes that the animal reaches its maximum
    theoretical speed at least once during the dataset.
    '''
    # NOTE possible problem: observed max could be outlier. outliers in
    # signal/obs could be filtered

    import numpy
    import scipy.integrate

    from bodycondition import utils

    speed     = numpy.zeros(len(rel_speed), dtype=float)
    speed_cal = numpy.zeros(len(rel_speed), dtype=float)

    # Get indices of data sections split by known zero velocities
    nonzero_mask = (rel_speed > zero_level) & (~numpy.isnan(rel_speed)) & \
                   (~numpy.isnan(acc_x))
    start_ind, stop_ind = utils.contiguous_regions(nonzero_mask)

    # Estimate speed in regions between known points of zero velocity
    for start_idx, stop_idx in zip(start_ind, stop_ind):
        y = acc_x[start_idx:stop_idx]
        # Make x in units of seconds for Hz # TODO TBC
        x = numpy.arange(0, len(y)) * fs_a

        # Integrate acceleration signal over periods where the relative
        # velocity sensor data is 'zero' velocity
        vel = scipy.integrate.cumtrapz(x, y, initial=0)

        # Assign total velocity to slice of acc_speed array
        speed[start_idx:stop_idx] = vel

    # Fit integrated accelerometer speed to relative speed sensor data
    measured    = rel_speed[:]
    calibration = speed#/fs_a #TODO remove, not correct

    # Fix for fitting data, otherwise got negative gain
    #calibration[calibration < 0] = 0

    # Generate input array as per numpy docs and apply least-squares regression
    # (i.e. generate coefficient matrix, each column term in f(x))
    A = numpy.vstack([measured, numpy.ones(len(measured))]).T
    gain, offset = numpy.linalg.lstsq(A, calibration)[0]

    if theoretic_max is not None:
        # Adjust gain for theoretical max speed of animal, assuming at least one
        # occurance of max speed
        gain = (theoretic_max-offset)/measured.max()

    # Apply calibration to relative speed sensor data
    # Do not apply offset when relative speed sensor is zero
    # TODO smooth this? creates a little bit jagged speeds
    mask = rel_speed > 0
    #speed_cal[mask] = (gain*measured[mask])+offset
    speed_cal = (gain*measured)+offset

    # Set values below zero to NaN if switch set
    if rm_neg is True:
        speed_cal[speed_cal < 0] = numpy.nan

    return speed_cal


