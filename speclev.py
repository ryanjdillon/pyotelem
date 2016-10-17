
def speclev(x, nfft=512, fs=1, w=None, nov=None):
    '''
    Args
    ----
    x:   signal from which the speclev (power spectra) is going to be calculated
    fft: fourier transform len
    fs:  frequency sample
    w:   if not specified then w is equal to the nfft
    nov: if not specified then nov is set to half the size of the nfft

    Attributes
    ----------
    S:   amount of power in each particular frequency (f)

    Returns
    -------
    SL:
    f:
    '''
    import numpy

    def buffer(x, n, p, opt='nodelay'):
        '''Mimic MATLAB routine to generate buffer array

        Args
        ----
        x:   signal array
        n:   number of data segments
        p:   number of values to overlap/underlap
        opt: initial condition options. default is to set the first `p` values
             to zero, while 'nodelay' to begin filling the buffer immediately.
        '''
        import numpy

        L = len(x)
        cols = int(numpy.ceil(L/(n-p)))

        b = numpy.zeros((cols, n))

        if p >= n:
            raise ValueError('`p` must be less than `n`.')

        if opt == 'nodelay':
            # skips initial condition, and begins filling buffer immediately
            if L < p:
                raise ValueError('p value too large for `nodelay`')

        else:
            raise SystemError('only `nodelay` implmented so far')

        for i in range(cols):
            print(i, p)
            if i != 0:
                # set first values of row to last p values
                b[i,:p] = b[i-1, -p:]
            else:
                b[i,:p] = 0

            b[i,p:] = x[n*i:(n*(i+1))-p]

        return b

    if w == None:
        w = nfft

    if nov == None:
        nov = nfft / 2

    if len(w) == 1:
        w = hanning(w)

    P = numpy.zeros((nfft / 2, x.shape[1]))

    for k in range(x.shape[1]):
        X, z = buffer(x[:, k], len(w), nov, 'nodelay')

        X = multiply(detrend(X), repmat(w, 1, X.shape[1]))

        F = abs(numpy.fft(X, nfft)) ** 2

        P[:, k] = sum(F[1:nfft / 2, :], 2)

    ndt = X.shape[1]

    # these two lines give correct output for randn input
    # SL of randn should be -10*log10(fs/2)
    slc = 3 - 10*log10(fs / nfft) - 10*log10(sum(w ** 2) / nfft)
    SL = 10*log10(P) - 10*log10(ndt) - 20*log10(nfft) + slc
    f = numpy.asarray(range(int((nfft/2)-1))) / nfft * fs

    return SL, f
