def buffer(x, n, p=0, opt=None, z_out=True):
    '''Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Args
    ----
    x:     Signal array
    n:     Number of data segments
    p:     Number of values to overlap
    opt:   Initial condition options. None (default) sets the first `p` values
           to zero, while 'nodelay' begins filling the buffer immediately.
    z_out: Boolean switch to return z array. True returns an additional array
           with these values. False returns only the buffer array including
           these values.

    Returns
    -------
    b:     buffer array with dimensions (n, cols)
    z:     array of values leftover that do not completely fill an n-length
           segment with overlap
    '''
    import numpy

    if p >= n:
        raise ValueError('p ({}) must be less than n ({}).'.format(p,n))

    # Calculate number of columns of buffer array
    cols = int(numpy.ceil(len(x)/(n-p)))

    # Check for opt parameters
    if opt == 'nodelay':
        # Need extra column to handle additional values left
        cols += 1
    elif opt != None:
        raise SystemError('Only `None` (default initial condition) and '
                          '`nodelay` (skip initial condition) have been '
                          'implemented')

    # Create empty buffer array
    b = numpy.zeros((n, cols))

    # Fill buffer by column handling for initial condition and overlap
    j = 0
    for i in range(cols):
        # Set first column to n values from x, move to next iteration
        if i == 0 and opt == 'nodelay':
            b[0:n,i] = x[0:n]
            continue
        # set first values of row to last p values
        elif i != 0 and p != 0:
            b[:p, i] = b[-p:, i-1]
        # If initial condition, set p elements in buffer array to zero
        else:
            b[:p, i] = 0

        # Get stop index positions for x
        k = j + n - p

        # Get stop index position for b, matching number sliced from x
        n_end = p+len(x[j:k])

        # Assign values to buffer array from x
        b[p:n_end,i] = x[j:k]

        # Update start index location for next iteration of x
        j = k

    # TODO implement this better
    # Hackish handling of z array creation. Problematic if redundant values;
    # though that should be very unlikely in sensor signal input, right?
    if (z_out == True):
        if any(b[:, -1] == 0):
            # make array of leftover elements without zeros or overlap repeats
            z = numpy.array([i for i in b[:,-1] if i != 0 and i not in b[:,-2]])
            b = b[:, :-1]
        else:
            z = numpy.array([])
        return b, z
    else:
        return b



def event_on(cue, t):
    '''Find indices where in t (time array) where event is on

    k = 1 when cue is on
    k = 0 when cue is off

    Args
    ----
    cue: numpy.ndarray, shape (n, 2)
        is a list of events in the format: cue = [start_time,duration]

    Returns
    -------
    k: numpy.ndarray, dtype bool
        index of cues where event is on
    not_k: numpy.ndarray, dtype bool
        is the complement of k.

    Note
    ----
    `notk` change to `k`
    '''
    import numpy

    k = numpy.zeros(len(t), dtype=bool)
    cst = cue[:, 0]
    ce = cue[:, 0] + cue[:, 1]

    for kk in range(len(cue)):
        k = k | ((t >= cst[kk]) & (t < ce[kk]))

    not_k = (k==0)

    return k, not_k


def contiguous_regions(condition):
    '''Finds contiguous True regions of the boolean array 'condition'.

    Args
    ----
    condition: numpy.ndarray, dtype bool
        boolean mask array, but can pass the condition itself (e.g. x > 5)

    Returns
    -------
    start: numpy.ndarray, dtype int
        array with the start indices for each contiguous region
    stop: numpy.ndarray, dtype int
        array with the stop indices for each contiguous region

    http://stackoverflow.com/a/4495197/943773
    '''
    import numpy

    if condition.ndim > 1:
        raise IndexError('contiguous_regions(): condition must be 1-D')

    # Find the indicies of changes in 'condition'
    idx = numpy.diff(condition).nonzero()[0]

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = numpy.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = numpy.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)

    # We need to start things after the change in 'condition'. Therefore,
    # we'll shift the index by 1 to the right.
    start = idx[:,0] + 1
    # keep the stop ending just before the change in condition
    stop  = idx[:,1]

    # remove reversed or same start/stop index
    good_vals = (stop-start) > 0
    start = start[good_vals]
    stop = stop[good_vals]

    return start, stop


def rm_regions(a, b, a_start_ind, a_stop_ind):
    '''Remove additional contiguous regions in `a` that occur before a
    complimentary region in `b` has occured'''
    import numpy

    for i in range(len(a_stop_ind)):
        next_a_start = numpy.argmax(a[a_stop_ind[i]:])
        next_b_start = numpy.argmax(b[a_stop_ind[i]:])
        if  next_b_start > next_a_start:
            a[a_start_ind[i]:a_stop_ind[i]] = False

        return a


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


