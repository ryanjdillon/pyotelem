
def normalized(a, axis=-1, order=2):
    '''Return normalized vector for arbitrary axis

    Args
    ----
    a: ndarray (n,3)
        Tri-axial vector data
    axis: int
        Axis index to overwhich to normalize
    order: int
        Order of nomalization to calculate

    Notes
    -----
    This function was adapted from the following StackOverflow answer:
    http://stackoverflow.com/a/21032099/943773
    '''
    import numpy

    l2 = numpy.atleast_1d(numpy.linalg.norm(a, order, axis))
    l2[l2==0] = 1

    return a / numpy.expand_dims(l2, axis)


def findzc(x, thresh, t_max=None):
    '''
    Find cues to each zero-crossing in vector x.

    To be accepted as a zero-crossing, the signal must pass from below
    -thresh to above thresh, or vice versa, in no more than t_max samples.

    Args
    ----
    thresh: (float)
        magnitude threshold for detecting a zero-crossing.
    t_max: (int)
        maximum duration in samples between threshold crossings.

    Returns
    -------
    zc: ndarray
        Array containing the start **zc_s**, finish **zc_f** and direction **S**
        of zero crossings

        where:

        * zc_s: the cue of the first threshold-crossing in samples
        * zc_f: the cue of the second threshold-crossing in samples
        * S: the sign of each zero-crossing (1 = positive-going, -1 = negative-going).

    Notes
    -----
    This routine is a reimplementation of Mark Johnson's Dtag toolbox method
    and tested against the Matlab version to be sure it has the same result.
    '''
    import numpy

    # positive threshold: p (over) n (under)
    pt_p = x > thresh
    pt_n = ~pt_p

    # negative threshold: p (over) n (under)
    nt_n = x < -thresh
    nt_p = ~nt_n

    # Over positive threshold +thresh
    # neg to pos
    pt_np = (pt_p[:-1] & pt_n[1:]).nonzero()[0]
    # pos to neg
    pt_pn = (pt_n[:-1] & pt_p[1:]).nonzero()[0] + 1

    # Over positive threshold +thresh
    # neg to pos
    nt_np = (nt_p[:-1] & nt_n[1:]).nonzero()[0] + 1
    # pos to neg
    nt_pn = (nt_n[:-1] & nt_p[1:]).nonzero()[0]

    # Concat indices, order sequentially
    ind_all = numpy.hstack((pt_np, nt_np, pt_pn, nt_pn))
    ind_all.sort()

    # Omit rows where just touching but not crossing
    crossing_mask = ~(numpy.diff(numpy.sign(x[ind_all])) == 0)

    # Append a False to make the same length as ind_all
    crossing_mask = numpy.hstack((crossing_mask, False))

    # Get 1st and 2nd crossings
    ind_1stx = ind_all[crossing_mask]
    ind_2ndx = ind_all[numpy.where(crossing_mask)[0]+1]

    # TODO odd option to replace with NaNs rather than delete?
    # Delete indices that do not have a second crossing
    del_ind = numpy.where(ind_2ndx > len(x)-1)[0]
    for i in del_ind:
        ind_1stx = numpy.delete(ind_1stx, i)
        ind_2ndx = numpy.delete(ind_1stx, i)

    # Get direction/sign of crossing
    signs = numpy.sign(x[ind_1stx])*-1

    # Add column of direction and transpose
    zc = numpy.vstack((ind_1stx, ind_2ndx, signs)).T

    # TODO not mentioned in docstring, remove?
    #x_norm? = ((x[:, 1] * zc[:, 0]) - (x[:, 0] * zc[:, 1])) / x[:, 1] - x[:, 0]

    if t_max:
        zc = zc[zc[:, 1] - zc[:, 0] <= t_max, :]

    return zc.astype(int)


def butter_filter(cutoff, fs, order=5, btype='low'):
    '''Create a digital butter fileter with cutoff frequency in Hz

    Args
    ----
    cutoff: float
        Cutoff frequency where filter should separate signals
    fs: float
        sampling frequency
    btype: str
        Type of filter type to create. 'low' creates a low-frequency filter and
        'high' creates a high-frequency filter (Default 'low).

    Returns
    -------
    b: ndarray
        Numerator polynomials of the IIR butter filter
    a: ndarray
        Denominator polynomials of the IIR butter filter

    Notes
    -----
    This function was adapted from the following StackOverflow answer:
    http://stackoverflow.com/a/25192640/943773
    '''
    import scipy.signal

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype=btype, analog=False)

    return b, a


def butter_apply(b, a, data):
    '''Apply filter with filtfilt to allign filtereted data with input

    The filter is applied once forward and once backward to give it linear
    phase, using Gustafsson's method to give the same length as the original
    signal.

    Args
    ----
    b: ndarray
        Numerator polynomials of the IIR butter filter
    a: ndarray
        Denominator polynomials of the IIR butter filter

    Returns
    -------
    x: ndarray
        Filtered data with linear phase

    Notes
    -----
    This function was adapted from the following StackOverflow answer:
    http://stackoverflow.com/a/25192640/943773
    '''
    import scipy.signal

    return scipy.signal.filtfilt(b, a, data, method='gust')


def calc_PSD_welch(x, fs, nperseg):
    '''Caclulate power spectral density with Welch's method

    Args
    ----
    x: ndarray
        sample array
    fs: float
        sampling frequency (1/dt)

    Returns
    -------
    f_welch: ndarray
        Discrete frequencies
    S_xx_welch: ndarray
        Estimated PSD at discrete frequencies `f_welch`
    P_welch: ndarray
        Signal power (integrated PSD)
    df_welch: ndarray
        Delta between discreet frequencies `f_welch`
    '''
    import numpy
    import scipy.signal

    # Code source and description of FFT, DFT, etc.
    # http://stackoverflow.com/a/33251324/943773
    dt = 1/fs
    N = len(x)
    times = numpy.arange(N) / fs

    # Estimate PSD `S_xx_welch` at discrete frequencies `f_welch`
    f_welch, S_xx_welch = scipy.signal.welch(x, fs=fs, nperseg=nperseg)

    # Integrate PSD over spectral bandwidth
    # to obtain signal power `P_welch`
    df_welch = f_welch[1] - f_welch[0]
    P_welch = numpy.sum(S_xx_welch) * df_welch

    return f_welch, S_xx_welch, P_welch, df_welch


def simple_peakfinder(x, y, delta):
    '''Detect local maxima and minima in a vector

    A point is considered a maximum peak if it has the maximal value, and was
    preceded (to the left) by a value lower by `delta`.

    Args
    ----
    y: ndarray
        array of values to find local maxima and minima in
    delta: float
        minimum change in `y` since previous peak to be considered new peak.
        It should be positive and it's absolute value taken to ensure this.
    x: ndarray
        corresponding x-axis positions to y array

    Returns
    -------
    max_ind: ndarray
        Indices of local maxima
    min_ind: ndarray
        Indices of local minima

    Example
    -------
    max_ind, min_ind = simple_peakfinder(x, y, delta)

    # get values of `y` at local maxima
    local_max = y[max_ind]

    Notes
    -----
    Matlab Author:      Eli Billauer   http://billauer.co.il/peakdet.html
    Python translation: Chris Muktar   https://gist.github.com/endolith/250860
    Python cleanup:     Ryan J. Dillon
    '''
    import numpy

    y = numpy.asarray(y)

    max_ind = list()
    min_ind = list()

    local_min     = numpy.inf
    local_max     = -numpy.inf
    local_min_pos = numpy.nan
    local_max_pos = numpy.nan

    lookformax = True

    for i in range(len(y)):

        if y[i] > local_max:
            local_max = y[i]
            local_max_pos = x[i]

        if y[i] < local_min:
            local_min = y[i]
            local_min_pos = x[i]

        if lookformax:
            if y[i] < local_max-abs(delta):
                max_ind.append(local_max_pos)
                local_min = y[i]
                local_min_pos = x[i]
                lookformax = False
        else:
            if y[i] > local_min+abs(delta):
                min_ind.append(local_min_pos)
                local_max = y[i]
                local_max_pos = x[i]
                lookformax = True

    return numpy.array(max_ind), numpy.array(min_ind)
