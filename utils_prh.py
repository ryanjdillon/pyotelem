
def calc_PRH(A_g):
    '''Calculate the pitch, roll and heading for triaxial movement signalsi

    Args
    ----
    A_g: numpy.ndarray, shape (n,3)
        triaxial movement array with n samples over 3 axes (0:x, 1:y, 2:z)

    Returns
    -------
    pitch: numpy.ndarray, 1D
        pitch in radians
    roll: numpy.ndarray, 1D
        roll in radians
    heading: numpy.ndarray, 1D
        heading in radians
    '''

    p = pitch(A_g[:,0], A_g[:,1], A_g[:,2])
    r  = roll(A_g[:,0], A_g[:,1], A_g[:,2])
    y   = yaw(A_g[:,0], A_g[:,1], A_g[:,2])

    #from dtag_toolbox_python.dtag2 import a2pr
    #pitch, roll, A_norm = a2pr.a2pr(A_g)

    return p, r, y


def pitch(x,y,z):
    '''Angle of x-axis relative to ground (theta)'''
    import numpy
    return numpy.arctan(x/numpy.sqrt(y**2+z**2))


def roll(x,y,z):
    '''Angle of y-axis relative to ground (phi)'''
    import numpy
    #return numpy.arctan(y/numpy.sqrt(x**2+z**2))
    return numpy.arctan2(y,z)


def yaw(x,y,z):
    '''Angle of z-axis relative to ground (psi)'''
    import numpy
    #return numpy.arctan(numpy.sqrt(x**2+y**2)/z)
    return numpy.arctan2(x,y)
