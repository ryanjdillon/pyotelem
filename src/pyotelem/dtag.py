def fix_gaps(y):
    """Linearly interpolates over NaNs in 1D input array x

    Args
    ----
    x: ndarray
        array containing NaNs to interpolate over

    Returns
    -------
    y_interp: ndarray
        array with NaN values replaced by interpolated values
    """
    import numpy

    # Boolean mask for all values not equal to NaN in input array
    not_nan = ~numpy.isnan(y)

    # x values for array elements to be interpolated
    xp = numpy.where(not_nan)

    # y values for array elements to be interpolated
    yp = y[not_nan]

    x_interp = numpy.arange(len(xp))
    y_interp = numpy.interp(x_interp, xp, yp)

    return y_interp


def buffer(x, n, p=0, opt=None, z_out=True):
    """Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Args
    ----
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. None (default) sets the first `p` values to
        zero, while 'nodelay' begins filling the buffer immediately.
    z_out: bool
        Boolean switch to return z array. True returns an additional array with
        these values. False returns only the buffer array including these
        values.

    Returns
    -------
    b: ndarray
        buffer array with dimensions (n, cols)
    z: ndarray
        array of values leftover that do not completely fill an n-length
        segment with overlap
    """
    import numpy

    if p >= n:
        raise ValueError("p ({}) must be less than n ({}).".format(p, n))

    # Calculate number of columns of buffer array
    cols = int(numpy.ceil(len(x) / (n - p)))

    # Check for opt parameters
    if opt == "nodelay":
        # Need extra column to handle additional values left
        cols += 1
    elif opt is not None:
        raise SystemError(
            "Only `None` (default initial condition) and "
            "`nodelay` (skip initial condition) have been "
            "implemented"
        )

    # Create empty buffer array
    b = numpy.zeros((n, cols))

    # Fill buffer by column handling for initial condition and overlap
    j = 0
    for i in range(cols):
        # Set first column to n values from x, move to next iteration
        if i == 0 and opt == "nodelay":
            b[0:n, i] = x[0:n]
            continue
        # set first values of row to last p values
        elif i != 0 and p != 0:
            b[:p, i] = b[-p:, i - 1]
        # If initial condition, set p elements in buffer array to zero
        else:
            b[:p, i] = 0

        # Get stop index positions for x
        k = j + n - p

        # Get stop index position for b, matching number sliced from x
        n_end = p + len(x[j:k])

        # Assign values to buffer array from x
        b[p:n_end, i] = x[j:k]

        # Update start index location for next iteration of x
        j = k

    # TODO implement this better
    # Hackish handling of z array creation. Problematic if redundant values;
    # though that should be very unlikely in sensor signal input, right?
    if z_out is True:
        if any(b[:, -1] == 0):
            # make array of leftover elements without zeros or overlap repeats
            z = numpy.array([i for i in b[:, -1] if i != 0 and i not in b[:, -2]])
            b = b[:, :-1]
        else:
            z = numpy.array([])
        return b, z
    else:
        return b


def event_on(cue, t):
    """Find indices where in t (time array) where event is on

    k = 1 when cue is on
    k = 0 when cue is off

    Args
    ----
    cue: numpy.ndarray, shape (n, 2)
        is a list of events in the format: cue = [start_time,duration]
    t: ndarray
        array of time stames in seconds

    Returns
    -------
    k: numpy.ndarray, dtype bool
        index of cues where event is on
    not_k: numpy.ndarray, dtype bool
        is the complement of k.

    Note
    ----
    `notk` change to `k`
    """
    import numpy

    k = numpy.zeros(len(t), dtype=bool)
    cst = cue[:, 0]
    ce = cue[:, 0] + cue[:, 1]

    for kk in range(len(cue)):
        k = k | ((t >= cst[kk]) & (t < ce[kk]))

    not_k = k == 0

    return k, not_k
