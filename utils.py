def recursive_input(input_label, type_class):
    '''General user input function

    Args
    ----
    type_class (type): name of python type (e.g. float, no parentheses)

    Returns
    -------
    output: value entered by user converted to type `type_class`
    '''
    msg = 'Enter {} {}: '.format(input_label, type_class)
    try:
        output = type_class(input(msg))
        return output
    except:
        return recursive_input(input_label, type_class)


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


def first_idx(condition):
    '''Return index of first occurance of true in boolean array'''
    import numpy

    return numpy.where(condition)[0][0]


def last_idx(condition):
    '''Return index of last occurance of true in boolean array
    '''
    import numpy

    return numpy.where(condition)[0][-1]


def normalized(a, axis=-1, order=2):
    '''Return normalized vector for arbitrary axis

    http://stackoverflow.com/a/21032099/943773
    '''
    import numpy

    l2 = numpy.atleast_1d(numpy.linalg.norm(a, order, axis))
    l2[l2==0] = 1

    return a / numpy.expand_dims(l2, axis)


