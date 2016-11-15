def load_experiments(root_path):
    '''Load experiment parameters from directory structure'''
    from collections import OrderedDict
    from pylleo.pylleo import yamlutils

    # TODO breaks if path not a lleo data path, reliant on consistend path
    # naming
    # /home/ryan/Desktop/edu/01_PhD/projects/smartmove/data/lleo_coexist/Acceleration

    exp_dict = OrderedDict()
    data_paths = list()

    for child in os.listdir(root_path):
        path = os.path.join(root_path, child)
        if os.path.isdir(path):
            child = child.strip()
            exp_dict[child] = OrderedDict()

            vals = child.split('_')
            exp_dict[child]['date']      = vals[0]
            exp_dict[child]['tag_model'] = vals[1].replace('-', '')
            exp_dict[child]['tag_id']    = vals[2]
            exp_dict[child]['animal']    = vals[3].lower()
            design = vals[4].lower()
            if design == 'control':
                exp_dict[child]['design'] = design
                exp_dict[child]['design_n'] = 0
            else:
                exp_dict[child]['design'] = int(design[0])
                exp_dict[child]['design_n'] = design[1:]

    return exp_dict


def normalized(a, axis=-1, order=2):
    '''Return normalized vector for arbitrary axis

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
    thresh: magnitude threshold for detecting a zero-crossing.
    t_max:  (optional) maximum duration in samples between threshold
            crossings.

    Returns
    -------
    K: nx3 matrix [Ks,Kf,S]
       where:
       * Ks contains the cue of the first threshold-crossing in samples
       * Kf contains the cue of the second threshold-crossing in samples
       * S contains the sign of each zero-crossing
         (1 = positive-going, -1 = negative-going).

    This routine is a reimplementation of Mark Johnson's dtag_toolbox method
    and tested against the Matlab version to be sure it has the same result.
    '''
    # positive threshold: p (over) n (under)
    pt_p = data > thresh
    pt_n = ~pt_p

    # negative threshold: p (over) n (under)
    nt_n = data < -thresh
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

    # Concat indices, order sequentially, the reshape to nx2 array
    ind = numpy.hstack((pt_np, nt_np, pt_pn, nt_pn))
    ind.sort()
    ind = ind.reshape((int(len(ind)/2), 2))

    # Omit rows where just touching but not crossing
    crossing_mask = ~(numpy.diff(numpy.sign(x[ind])).ravel() == 0)
    ind = ind[crossing_mask]

    # Add column of direction of crossing (-1: neg to pos, 1: pos to neg)
    # Need transpose ind array, stack, then transpose back for some reason
    K = numpy.vstack((ind.T, numpy.sign(x[ind])[:,1])).T

    # TODO not mentioned in docstring, remove?
    #x_norm? = ((X[:, 1] * K[:, 0]) - (X[:, 0] * K[:, 1])) / X[:, 1] - X[:, 0]

    if t_max:
        K = K[K[:, 1] - K[:, 0] <= t_max, :]

    return K


def peakfinder(x, sel=None, thresh=None, extrema=None):
    '''
    PEAKFINDER Noise tolerant fast peak finding algorithm

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html

    Args
    ----
    x0:      A real vector from the maxima will be found (required)

    sel:     The amount above surrounding data for a peak to be identified
             (default = (max(x0)-min(x0))/4). Larger values mean the algorithm
             is more selective in finding peaks.

    thresh:  A threshold value which peaks must be larger than to be maxima or
             smaller than to be minima.

    extrema: 1 if maxima are desired, -1 if minima are desired
             (default = maxima, 1)

    Returns
    -------
    peakLoc: The indicies of the identified peaks in x0

    peakMag: The magnitude of the identified peaks

    [peakLoc] = peakfinder(x0) returns the indicies of local maxima that
        are at least 1/4 the range of the data above surrounding data.

    [peakLoc] = peakfinder(x0,sel) returns the indicies of local maxima
        that are at least sel (default 1/4 the range of the data) above
        surrounding data.


    Note: If repeated values are found the first is identified as the peak

    Ex:
    peak_loc, peak_mag = peakfinder(x0,sel,thresh)

    returns the indicies and magnitude of local maxima that are at least sel
    above surrounding data and larger (smaller) than thresh if you are finding
    maxima (minima). returns the maxima of the data if extrema > 0 and the
    minima of the data if extrema < 0

    t = 0:.0001:10;
    x = 12*sin(10*2*pi*t)-3*sin(.1*2*pi*t)+randn(1,numel(t));
    x(1250:1255) = max(x);
    peakfinder(x)

    Implemented to take the same arguments and return the same output as the
    peakfinder.m routine written by Nathanael C. Yoder 2011 (nyoder@gmail.com)
    '''
    import numpy
    import scipy.signal

    # TODO belive this should be inversed, look into
    if sel == None:
        sel = (max(x)-min(x))/4

    # TODO handle extrema arg for returning minima

    # 1D data array
    vector = x
    # expected sample widths in signal to contain peaks
    widths = numpy.array([sel])
    wavelet = None
    max_distances = None
    # default 2
    gap_thresh = None
    min_length = None
    min_snr = 1
    noise_perc = 10

    peak_loc = numpy.asarray(scipy.signal.find_peaks_cwt(y, widths))
    peak_loc = peak_loc[y[peak_loc]>thresh]
    peak_mag = y[peak_loc]

    return peak_loc, peak_mag


def fir_nodelay(x, n, fp, qual=None):
    '''
    Generate a filter by call to numpy.signal.firwin()

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
    y = scipy.signal.lfilter(b, numpy_ones(len(b)), x)

    return y, b
