'''
Signal processing plotting functions
'''
import matplotlib.pyplot as plt

from . import plotconfig as _plotconfig
from .plotconfig import _colors, _linewidth


def plot_lf_hf(x, xlf, xhf, title=''):
    '''Plot original signal, low-pass filtered, and high-pass filtered signals

    Args
    ----
    x: ndarray
        Signal data array
    xlf: ndarray
        Low-pass filtered signal
    xhf: ndarray
        High-pass filtered signal
    title: str
        Main title of plot
    '''
    from . import plotutils

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, sharey=True)

    plt.title(title)

    #ax1.title.set_text('All freqencies')
    ax1.plot(range(len(x)), x, color=_colors[0], linewidth=_linewidth,
             label='original')
    ax1.legend(loc='upper right')

    #ax2.title.set_text('Low-pass filtered')
    ax2.plot(range(len(xlf)), xlf, color=_colors[1], linewidth=_linewidth,
             label='low-pass')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Frequency (Hz)')

    #ax3.title.set_text('High-pass filtered')
    ax3.plot(range(len(xhf)), xhf, color=_colors[2], linewidth=_linewidth,
             label='high-pass')
    ax3.legend(loc='upper right')


    ax1, ax2, ax3 = plotutils.add_alpha_labels([ax1, ax2, ax3], color='black',
                                               facecolor='none',
                                               edgecolor='none')

    # TODO break into util function
    # Convert sample # ticks to times
    total_seconds = ax3.get_xticks()/16

    # with timedelta: (stop - start).total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    strtime = lambda minutes, seconds: '{:.0f}:{:02.0f}'.format(minutes, seconds)
    labels = list(map(strtime, minutes, seconds))

    ax3.set_xticklabels(labels)

    plt.xlabel('Sample time (minute:second)')
    plt.show()

    return None


def plot_acc_pry_depth(A_g_lf, A_g_hf, pry_deg, depths, glide_mask=None):
    '''Plot the acceleration with the pitch, roll, and heading

    Args
    ----
    A_g_lf: ndarray
        Low-pass filtered calibration accelerometer signal
    A_g_hf: ndarray
        High-pass filtered calibration accelerometer signal
    pry_deg: ndarray
        Pitch roll and heading in degrees
    depths: ndarray
        Depth data for all samples
    glide_mask: ndarray
        Boolean array for slicing glides from tag data
    '''

    import numpy

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)

    ax1.title.set_text('acceleromter')
    ax1.plot(range(len(A_g_lf)), A_g_lf, color=_colors[0])

    ax1.title.set_text('PRH')
    ax2.plot(range(len(pry_deg)), pry_deg, color=_colors[1])

    ax3.title.set_text('depth')
    ax3.plot(range(len(depths)), depths, color=_colors[2])
    ax3.invert_yaxis()

    if glide_mask is not None:
        ind = range(len(glide_mask))
        ax1 = plot_shade_mask(ax1, ind, glide_mask)
        ax2 = plot_shade_mask(ax2, ind, glide_mask)
        ax3 = plot_shade_mask(ax3, ind, glide_mask)

    plt.show()

    return None


def plot_welch_peaks(f, S, peak_loc=None, title=''):
    '''Plot welch PSD with peaks as scatter points

    Args
    ----
    f: ndarray
        Array of frequencies produced with PSD
    S: ndarray
        Array of powers produced with PSD
    peak_loc: ndarray
        Indices of peak locations in signal
    title: str
        Main title for plot
    '''
    plt.plot(f, S, linewidth=_linewidth)
    plt.title(title)
    plt.xlabel('Fequency (Hz)')
    plt.ylabel('"Power" (g**2 Hz**−1)')

    if peak_loc is not None:
        plt.scatter(f[peak_loc], S[peak_loc], label='peaks')
        plt.legend(loc='upper right')

    plt.show()

    return None


def plot_fft(f, S, dt):
    '''Plot fft

    Args
    ----
    f: ndarray
        Array of frequencies produced with PSD
    S: ndarray
        Array of powers produced with PSD
    dt: ndarray
        Sampling rate of sensor
    '''
    import numpy

    xf = numpy.linspace(0.0, 1/(2.0*dt), N/2)
    plt.plot(xf, 2.0/N * numpy.abs(S[:N//2]), linewidth=_linewidth)
    plt.show()

    return None


def plot_welch_perdiogram(x, fs, nperseg):
    '''Plot Welch perdiogram

    Args
    ----
    x: ndarray
        Signal array
    fs: float
        Sampling frequency
    nperseg: float
        Length of each data segment in PSD
    '''
    import scipy.signal
    import numpy

    # Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
    # 0.001V**2/Hz of white noise sampled at 10 kHz.
    N = len(x)
    time = numpy.arange(N) / fs

    # Compute and plot the power spectral density.
    f, Pxx_den = scipy.signal.welch(x, fs, nperseg=nperseg)

    plt.semilogy(f, Pxx_den)
    plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

    # If we average the last half of the spectral density, to exclude the peak,
    # we can recover the noise power on the signal.
    numpy.mean(Pxx_den[256:])  # 0.0009924865443739191

    # compute power spectrum
    f, Pxx_spec = scipy.signal.welch(x, fs, 'flattop', 1024,
                                     scaling='spectrum')

    plt.figure()
    plt.semilogy(f, numpy.sqrt(Pxx_spec))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.show()

    return None


def plot_data_filter(data, data_f, b, a, cutoff, fs):
    '''Plot frequency response and filter overlay for butter filtered data

    Args
    ----
    data: ndarray
        Signal array
    data_f: float
        Signal sampling rate
    b: array_like
        Numerator of a linear filter
    a: array_like
        Denominator of a linear filter
    cutoff: float
        Cutoff frequency for the filter
    fs: float
        Sampling rate of the signal

    Notes
    -----
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
    fig, (ax1, ax2) = plt.subplots(2,1)

    ax1.title.set_text('Lowpass Filter Frequency Response')
    ax1.plot(0.5*fs * w/numpy.pi, numpy.abs(h), 'b')
    ax1.plot(cutoff, 0.5*numpy.sqrt(2), 'ko')
    ax1.axvline(cutoff, color='k')
    ax1.set_xlim(0, 0.5*fs)
    ax1.set_xlabel('Frequency [Hz]')
    ax2.legend()

    # Demonstrate the use of the filter.
    ax2.plot(t, data, linewidth=_linewidth, label='data')
    ax2.plot(t, data_f, linewidth=_linewidth, label='filtered data')
    ax2.set_xlabel('Time [sec]')
    ax2.legend()

    plt.show()

    return None


