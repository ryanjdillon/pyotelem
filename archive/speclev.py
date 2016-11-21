
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


def speclev(x, nfft=512, fs=1, w=None, nov=None):
    '''
    Args
    ----
    x:    signal from which the speclev (power spectra) is to be calculated
    nfft: fourier transform len
    fs:   frequency sample
    w:    if not specified then w is equal to the nfft
    nov:  if not specified then nov is set to half the size of the nfft

    Attributes
    ----------
    S:   amount of power in each particular frequency (f)

    Returns
    -------
    SL:
    f:
    '''
    import numpy
    import scipy.signal

    if w == None:
        w = nfft

    if nov == None:
        nov = nfft / 2

    if len(w) == 1:
        w = hanning(w)

    P = numpy.zeros((nfft / 2, x.shape[1]))

    for k in range(len(x.shape[1])):
        X, z = buffer(x[:, k], len(w), nov, 'nodelay', z_out=True)

        X = scipy.signal.detrend(X) * w

        F = numpy.abs(numpy.fft(X, nfft)) ** 2
        P[:, k] = numpy.sum(F[:int(nfft/2), :], axis=1)

    ndt = X.shape[1]

    # these two lines give correct output for randn input
    # SL of randn should be -10*log10(fs/2)
    slc = 3 - 10*log10(fs / nfft) - 10*log10(sum(w ** 2) / nfft)
    SL = 10*log10(P) - 10*log10(ndt) - 20*log10(nfft) + slc
    f = numpy.asarray(range(int((nfft/2)-1))) / nfft * fs

    return SL, f
