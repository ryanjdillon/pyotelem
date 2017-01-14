
def calc_PRH(ax, ay, az):
    '''Calculate the pitch, roll and heading for triaxial movement signalsi

    Args
    ----
    A_g: numpy.ndarray, shape (n,3)
        triaxial movement array with n samples over 3 axes (0:ax, 1:ay, 2:az)

    Returns
    -------
    pitch: numpy.ndarray, 1D
        pitch in radians
    roll: numpy.ndarray, 1D
        roll in radians
    heading: numpy.ndarray, 1D
        heading in radians
    '''

    pitch = calc_pitch(ax, ay, az)
    roll  = calc_roll(ax, ay, az)
    yaw   = calc_yaw(ax, ay, az)

    #from dtag_toolbox_python.dtag2 import a2pr
    #pitch, roll, A_norm = a2pr.a2pr(A_g)

    return pitch, roll, yaw


def calc_pitch(ax, ay, az):
    '''Angle of x-axis relative to ground (theta)'''
    import numpy
    # arctan2 not needed here to cover all quadrants, just for consistency
    return numpy.arctan(ax, numpy.sqrt(ay**2+az**2))


def calc_roll(ax, ay, az):
    '''Angle of y-axis relative to ground (phi)'''
    import numpy
    #return numpy.arctan(ay/numpy.sqrt(ax**2+az**2))
    return numpy.arctan2(ay,az)


def calc_yaw(ax, ay, az):
    '''Angle of z-axis relative to ground (psi)'''
    import numpy
    #return numpy.arctan(numpy.sqrt(ax**2+ay**2)/az)
    return numpy.arctan2(ax,ay)


def absdeg(deg):
    '''Change from signed degrees to 0-180 or 0-360 ranges

    e.g. An array with values -180:180 becomes 0:360
    e.g. An array with values -90:90 becomes 0:180
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

    http://physics.stackexchange.com/a/41655/126878
    '''
    import numpy
    return numpy.sqrt(ax**2 + ay**2 + az**2)


def triaxial_integration(x, y, z, initial=0):
    '''Integrate three axes to obtain velocity of position'''
    import scipy.integrate

    x_int = scipy.integrate.cumtrapz(range(len(x)), x, initial=initial)
    y_int = scipy.integrate.cumtrapz(range(len(y)), y, initial=initial)
    z_int = scipy.integrate.cumtrapz(range(len(z)), z, initial=initial)

    return x_int, y_int, z_int


def speed_from_acc_and_ref(x, fs_a, rel_speed, zero_level, theoretic_max=None,
        rm_neg=True):
    '''Estimate speed from x-axis acceleration, fitting to relative speed data

    Args
    ----
    x: ndarray
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
    x_speed_cal: ndarray
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

    import utils

    x_speed = numpy.zeros(len(rel_speed), dtype=float)

    # Get indices of data sections split by known zero velocities
    zero_mask = rel_speed > zero_level
    start_ind, stop_ind = utils.contiguous_regions(zero_mask)

    # Estimate speed in regions between known points of zero velocity
    for start_idx, stop_idx in zip(start_ind, stop_ind):
        xi = x[start_idx:stop_idx]

        # Integrate acceleration signal over periods where the relative
        # velocity sensor data is 'zero' velocity
        x_int = scipy.integrate.cumtrapz(range(len(xi)), xi, initial=0)

        # Assign total velocity to slice of acc_speed array
        x_speed[start_idx:stop_idx] = x_int

    # Fit integrated accelerometer speed to relative speed sensor data
    measured = rel_speed[:]
    calibration = x_speed/fs_a

    # Generate input array as per numpy docs and apply least-squares regression
    A = numpy.vstack([measured, numpy.ones(len(measured))]).T
    gain, offset = numpy.linalg.lstsq(A, calibration)[0]

    if theoretic_max is not None:
        # Adjust gain for theoretical max speed of animal, assuming at least one
        # occurance of max speed
        gain = (theoretic_max-offset)/measured.max()

    # Apply calibration to relative speed sensor data
    x_speed_cal = (measured*gain)+offset

    # Set values below zero to NaN if switch set
    if rm_neg is True:
        x_speed_cal[x_speed_cal < 0] = numpy.nan

    return x_speed_cal

