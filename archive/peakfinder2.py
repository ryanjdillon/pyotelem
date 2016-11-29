
def peakfinder(x0, sel=None, thresh=None, extrema=None):
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
