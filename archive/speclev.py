
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
