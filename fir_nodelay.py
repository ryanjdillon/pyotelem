
def fir_nodelay(x, n, fp, qual=None):
    '''
    Generate a filter by call to fir1()

    Args
    ----
    n:    length of symmetric FIR filter to use.
    fp:   filter cut-off frequency relative to fs/2=1
    qual: optional qualifier to pass to fir1.

    Returns
    -------
    y: filtered data?
    h: the filter used.

    For information on applying FIR filters, see the following link:
    http://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html
    '''
    import numpy
    import scipy.signal

    # TODO check scipy.signal against .m code, remove commented code

    # The filter is generated by a call to fir1:
    if qual == None:
        #h = fir1(n, fp)
        b = scipy.signal.firwin(n, fp)
    else:
        #h = fir1(n, fp, qual)
        b = scipy.signal.firwin(n, fp, window=qual)

    #n_offs = numpy.floor(n / 2)

    #if x.shape[0] == 1:
    #    x = ravel(x)

    #temp = numpy.hstack(([x[n:-1:2, :]],
    #                     [x],
    #                     [x[end() + range(- 1, - n, - 1), :]] ))
    #y = filter(h, 1, temp)
    #
    #y = y[n + n_offs - 1 + range(len(x)), :]

    # For FIR filter, use `1` for all denominator coefficients
        b = scipy.signal.firwin(n, fp)
    y = scipy.signal.lfilter(b, numpy_ones(len(b)), x)

    return y, b