
# types of filters:
#  - FFT
#  - Wiener filter (attenuated noise > signal, amplified signal > noise)
#  - wavelet denoising - decomposed to wavelets, small coefficients zeroed. works
#    well due to multi-resolution nature of wavelets, signal processed separately
#    in freq bands defined by the wavelet transform.
#     * see Haar wavelet SWT denoising

def inst_speed(depths, smoothpitch, fs, stroke_f, f, ind, thresh_deg):
    '''Estimate instantaneous swimming speed as depthrate

    Args
    ----
    depths:
        the depth time series in meters, sampled at fs Hz.
    smoothpitch:
        pitch estimated from the low pass filtered acceleration signal
    fs:
        sensor sampling rate in Hz
    stroke_f:
        equal to the nominal stroking rate in Hz. Default value is 0.5
        Hz. Use [] to get the default value if you want to enter
        subsequent arguments.
    f:
        number that multiplied by the stroke_f gives the cut-off frequency Wn,
        of the low pass filter. f is a fraction of stroke_f e.g., 0.4.
    ind:
        sample range over which to analyse.
    thresh_deg:
        degrees threshold for which the speed will be estimated

    Returns
    -------
    inst_speed:
        instantaneous speed calculated as the depth rate.

    Notes
    -----
    `p`  changed to `depths`
    `fl` changed to `Wn`
    `k`  changed to `ind`
    `Instspeed` changed to `inst_speed`
    `SwimSp` changed to `swim_speed`

    Lucia Martina Martin Lopez (May 2016)
    lmml2@st-andrews.ac.uk
    '''
    import numpy

    import utils_signal

    # TODO default stroke_f rate 0.5 here, 0.4 elsewhere

    # define the cut off frequency (cutoff) for the low-pass filter
    cutoff = f * stroke_f

    # Wn is the filter cut-off normalized to half the sampling frequency
    # (a.k.a the critical frequency).
    nyq = 0.5 * fs
    Wn  = cutoff / nyq

    # define the length of symmetric FIR (Finite Impulse Response) filter.
    n_f = round((fs / cutoff) * 4)

    # apply a symmetric FIR low-pass filter to Aw and Mw with 0 group delay to
    # obtain the low-pass filtered acceleration signal Alf
    speed = numpy.diff(depths[ind]) * fs
    speed_filtered, speed_filter = utils_signal.fir_nodelay(speed, n_f, Wn)

    InstSpeed  = numpy.hstack((speed_filtered, numpy.nan))

    swim_speed = -InstSpeed / smoothpitch
    xa         = numpy.copy(smoothpitch)
    swim_speed[abs(numpy.rad2deg(xa)) < thresh_deg] = numpy.nan

    return swim_speed


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
    zc: nx3 matrix [zc_s,zc_f,S]
        array containing the start, finish and direction of zero crossings

        where:

        * zc_s contains the cue of the first threshold-crossing in samples
        * zc_f contains the cue of the second threshold-crossing in samples
        * S contains the sign of each zero-crossing
          (1 = positive-going, -1 = negative-going).

    This routine is a reimplementation of Mark Johnson's dtag_toolbox method
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


def runmean(x, n):
    '''Perform running mean on x with window width n using numpy convolve

    http://stackoverflow.com/a/22621523/943773
    '''
    import numpy

    return numpy.convolve(x, numpy.ones((n,))/n, mode='valid')


def blackman_sinc(data, fc, b, mode='low'):
    '''Apply Blackman windowed-sinc low-pass filter to tri-axial acc data

    https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter

    fc : Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b : Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    '''

    import numpy

    # Compute filter length N (must be odd)
    N = int(numpy.ceil((4/b)))
    if not N % 2: N += 1
    n = numpy.arange(N)

    # Compute sinc filter
    h = numpy.sinc(2*fc*(n-(N-1)/2.))

    # Compute Blackman window
    w = 0.42 - 0.5 * numpy.cos(2*numpy.pi*n/(N-1)) + \
        0.08 * numpy.cos(4*numpy.pi*n/(N-1))
    #w = numpy.blackman(N)

    # Multiply sinc filter with window
    h = h * w

    # Normalize to get unity gain
    h = h / numpy.sum(h)

    if mode=='high':
        # Create a high-pass filter from the low-pass filter through spectral inversion
        h = -h
        h[(N-1)/2] += 1

    return numpy.convolve(data, h)


def wavelet_denoise(data, wavelet, noise_sigma):
    '''Filter 1D data using wavelet denoising

    Source: https://goo.gl/gOQwy5
    '''
    import numpy
    import scipy
    import pywt

    from pywt import dwt_max_level

    wavelet = pywt.Wavelet(wavelet)

    # Link above suggest this method, but it evaluates to levels `pywt` finds
    # to be higher than that determined by pywt.dwt_max_level(data_len, filter_len)
    # numpy.floor(numpy.log2(data.shape[0]))).astype(int))

    # use wavedec2 for 2d data
    wavelet_coeffs = pywt.wavedec(data, wavelet, level=None)
    threshold = noise_sigma*numpy.sqrt(2*numpy.log2(data.size))

    new_wavelet_coeffs = map(lambda x: pywt.threshold(x, threshold, mode='soft'),
                             wavelet_coeffs)

    return pywt.waverec(list(new_wavelet_coeffs), wavelet)


def butter_filter(cutoff, fs, order=5, btype='low'):
    '''Create a digital butter fileter with cutoff frequency in Hz

    # http://stackoverflow.com/a/25192640/943773
    '''
    import scipy.signal

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)

    return b, a


def butter_apply(b, a, data, method='gust'):
    '''Apply filter with filtfilt to allign filtereted data with input

    http://stackoverflow.com/a/25192640/943773
    '''
    import scipy.signal
    #y = lfilter(b, a, data)
    return scipy.signal.filtfilt(b, a, data, method='gust')


def plot_data_filter(data, data_f, b, a, cutoff, fs):
    '''Plot frequency response and filter overlay for butter filtered data

    http://stackoverflow.com/a/25192640/943773
    '''
    import matplotlib.pyplot as plt
    import numpy
    import scipy.signal

    n = len(data)
    T = n/fs
    t = numpy.linspace(0, T, n, endpoint=False)

    # Calculate frequency response
    w, h = scipy.signal.freqz(b, a, worN=8000)

    # Plot the frequency response.
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs * w/numpy.pi, numpy.abs(h), 'b')
    plt.plot(cutoff, 0.5*numpy.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)

    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    # Demonstrate the use of the filter.
    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, data_f, 'g-', linewidth=2, label='filtered data')

    plt.xlabel('Time [sec]')
    plt.grid()

    plt.legend()
    plt.subplots_adjust(hspace=0.35)
    plt.show()

    return None


def __test_data():
    '''Generate test data for testing filters'''
    import numpy

    # First make some data to be filtered.
    fs = 30.0       # sample rate, Hz
    T = 5.0         # seconds
    n = int(T * fs) # total number of samples
    t = numpy.linspace(0, T, n, endpoint=False)

    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = numpy.sin(1.2*2*numpy.pi*t) + 1.5*numpy.cos(9*2*numpy.pi*t) + \
           0.5*numpy.sin(12.0*2*numpy.pi*t)

    return data, fs


def calc_PSD_welch(x, fs, nperseg):
    '''Caclulate power spectral density with Welch's method

    Args
    ----
    x:        sample array
    fs:       sampling frequency (1/dt)

    Returns
    -------
    f_welch:    Discrete frequencies
    S_xx_welch: Estimated PSD at discrete frequencies `f_welch`
    P_welch:    Signal power (integrated PSD)
    df_welch:   Delta between discreet frequencies `f_welch`
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


def calc_PSD_fft(x, fs):
    '''Caclulate power spectral density with Fourier Transform

    Args
    ----
    x:        sample array
    fs:       sampling frequency (1/dt)

    Returns
    -------
    Xk:       Discrete Fourier Transform (DFT)
    f_fft:    Discrete frequencies
    S_xx_fft: Estimated PSD at discrete frequencies `f_fft`
    P_fft:    Signal power (integrated PSD)
    df_fft:   Delta between discreet frequencies `f_fft`
    '''
    import numpy

    N = len(x)
    dt = 1/fs
    time = numpy.arange(N) / fs

    # Compute DFT
    Xk = numpy.fft.fft(x)

    # Compute corresponding frequencies
    f_fft = numpy.fft.fftfreq(len(x), d=dt)

    # Estimate PSD `S_xx_fft` at discrete frequencies `f_fft`
    T = time[-1] - time[0]
    S_xx_fft = ((numpy.abs(Xk) * dt) ** 2) / T

    # Integrate PSD over spectral bandwidth to obtain signal power `P_fft`
    df_fft = f_fft[1] - f_fft[0]
    P_fft = numpy.sum(S_xx_fft) * df_fft

    return Xk, f_fft, S_xx_fft, P_fft, df_fft


def calc_E_fft_welch(x, Xk, fs, df_welch, df_fft, S_xx_welch):
    '''Calculate signal energy using discrete version of Parseval's theorem

    E = E_fft is the statement of the discrete version of Parseval's theorem

    Returns
    -------
    E_fft: energy from integrated DFT components over frequency
    E_welch: signal energy from Welch's PSD
    '''
    N = len(x)
    dt = 1/fs

    # Energy obtained via "integrating" over time
    E = numpy.sum(x ** 2)

    # Energy obtained via "integrating" DFT components over frequency.
    # The fact that `E` = `E_fft` is the statement of the discrete version of
    # Parseval's theorem.
    E_fft = numpy.sum(numpy.abs(Xk) ** 2) / N

    # Signal energy from Welch's PSD
    E_welch = (1. / dt) * (df_welch / df_fft) * numpy.sum(S_xx_welch)

    return E_fft, E_welch


def simple_fft(x):
    import scipy.fftpack

    S_fft = scipy.fftpack.fft(x)

    return S_fft


def peakfinder(x, sel=None, peak_thresh=0, extrema=None):
    '''
    PEAKFINDER Noise tolerant fast peak finding algorithm

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html

    Args
    ----
    x0: (1-D ndarray)
        A real vector from the maxima will be found (required)

    sel: (float)
        The amount above surrounding data for a peak to be identified
        (default = (max(x0)-min(x0))/4). Larger values mean the algorithm is
        more selective in finding peaks.

    peak_thresh: (float)
        A threshold value which peaks must be larger than to be maxima or
        smaller than to be minima.

    extrema: (integer)
        1 if maxima are desired -1 if minima are desired (default = maxima, 1)

    Returns
    -------
    peakLoc: (1-D ndarray)
        The indicies of the identified peaks in x0

    peakMag: (ndarray)
        The magnitude of the identified peaks

    Examples
    --------
    .. note: If repeated values are found the first is identified as the peak

    Return the indicies and magnitude of local maxima that are at least sel
    above surrounding data and larger (smaller) than thresh if you are finding
    maxima (minima). returns the maxima of the data if extrema > 0 and the
    minima of the data if extrema < 0::

        peak_loc, peak_mag = peakfinder(x0,sel,peak_thresh)

    Implemented to take the same arguments and return the same output as the
    peakfinder.m routine written by Nathanael C. Yoder 2011 (nyoder@gmail.com)
    '''
    import numpy
    import scipy.signal

    if sel == None:
        sel = (max(x)-min(x))/4

    # TODO handle extrema arg for returning minima

    # find_peaks_cwt() parameter defaut values
    # expected sample widths in signal to contain peaks
    widths = numpy.array([sel])
    wavelet = None
    max_distances = None
    gap_thresh = None # default 2
    min_length = None
    min_snr = 1
    noise_perc = 10

    peak_loc = numpy.asarray(scipy.signal.find_peaks_cwt(x, widths))

    # If no peaks found, empty array returned
    if peak_loc.size == 0:
        raise IndexError('No peak locations found')

    peak_loc = peak_loc[x[peak_loc]>peak_thresh]
    peak_mag = x[peak_loc]

    return peak_loc, peak_mag

# TODO finish test
#def test_peakfinder():
#    t = 0:.0001:10;
#    x = 12*sin(10*2*pi*t)-3*sin(.1*2*pi*t)+randn(1,numel(t));
#    x(1250:1255) = max(x);
#
#    peak_loc, peak_mag = peakfinder(x)
#
#    assert peak_loc = 


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
    y = scipy.signal.lfilter(b, numpy.ones(len(b)), x)

    return y, b


def __test_butter_filter(data, fs):
    '''Test butter filters'''

    # Filter requirements.
    order = 6
    cutoff = 3.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Filter the data, and plot both the original and filtered signals.
    data_f = butter_apply(b, a, data)
    plot_data_filter(data, data_f, b, a, cutoff, fs)

    return None


def __test_wavelet_filter(data):
    '''Test wavelet filters'''

    # Wavelet denoising
    noise_sigma = 16.0

    # many wave forms available: pywt.wavelist()
    wavelet = 'db6'

    yf = wavelet_denoise(data, wavelet, noise_sigma)

    return None


if __name__ == '__main__':

    data, fs = __test_data()
    __test_butter_filter(data, fs)
    __test_wavelet_filter(data)
