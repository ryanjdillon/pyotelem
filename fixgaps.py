
def fixgaps(y):
    '''Linearly interpolates over NaNs in 1D input array x

    Args
    ----
    x: array containing NaNs to interpolate over
    Returns
    -------
    y_interp: array with NaN values replaced by interpolated values
    '''
    import numpy

    # Boolean mask for all values not equal to NaN in input array
    not_nan = ~numpy.isnan(y)

    # x values for array elements to be interpolated
    xp = numpy.where(not_nan)

    # y values for array elements to be interpolated
    yp = y[not_nan]

    x_interp = numpy.arange(len(x))
    y_interp = numpy.interp(x_interp, xp, yp)

    return y_interp
