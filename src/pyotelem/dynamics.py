
def prh(ax, ay, az):
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

    p = pitch(ax, ay, az)
    r = roll(ax, ay, az)
    y = yaw(ax, ay, az)

    #from dtag_toolbox_python.dtag2 import a2pr
    #pitch, roll, A_norm = a2pr.a2pr(A_g)

    return p, r, y


def pitch(ax, ay, az):
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


def roll(ax, ay, az):
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


def yaw(ax, ay, az):
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
