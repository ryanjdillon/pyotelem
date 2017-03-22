import matplotlib.pyplot as plt
import seaborn

# Use specified style (e.g. 'ggplot')
plt.style.use('seaborn-whitegrid')

# Use specified color palette
colors = seaborn.color_palette()

# Global axis properties
linewidth = 0.5


# Data Exploration
#------------------------------------------------------------------------------
def compare_sensors(param, singles=False, cfg_paths_fname='./cfg_paths.yaml'):
    '''Compare data accross data files in acclerometer data folder
    '''
    import matplotlib.pyplot as plt
    import os
    import pandas
    import seaborn

    from rjdtools import yaml_tools

    cfg_paths = yaml_tools.read_yaml(cfg_paths_fname)
    root_data = os.path.join(cfg_paths['root'], cfg_paths['acc'])

    c = 0
    # Ugly color palette for us color-blind people, could be improved
    dir_list = sorted(os.listdir(root_data))
    colors = seaborn.color_palette("Paired", len(dir_list))
    for d in dir_list:
        if os.path.isdir(d):
            for f in os.listdir(os.path.join(root_data, d)):
                if f.startswith('pydata'):
                    fname = os.path.join(root_data, d, f)
                    data = pandas.read_pickle(fname)
                    plt.plot(data[param], label=f, color=colors[c])
                    c += 1
                    if singles:
                        plt.legend()
                        plt.show()
    if not singles:
        plt.legend()
        plt.show()

    return None

# Utils
#------------------------------------------------------------------------------

def merge_limits(axes, xlim=True, ylim=True):
    '''Set maximum and minimum limits from list of axis objects to each axis

    Args
    ----
    axes: iterable
        list of `matplotlib.pyplot` axis objects whose limits should be modified
    xlim: bool
        Flag to set modification of x axis limits
    ylim: bool
        Flag to set modification of y axis limits
    '''

    # Compile lists of all x/y limits
    xlims = list()
    ylims = list()
    for ax in axes:
        [xlims.append(lim) for lim in ax.get_xlim()]
        [ylims.append(lim) for lim in ax.get_ylim()]

    # Iterate over axes objects and set limits
    for ax in axes:
        if xlim:
            ax.set_xlim(min(xlims), max(xlims))
        if ylim:
            ax.set_ylim(min(ylims), max(ylims))

    return None


def plot_noncontiguous(ax, data, ind, color=colors[0], label=''):
    '''Plot non-contiguous slice of data

    Args
    ----
    data: 1-D numpy array
    ind: indices of data to be plotted

    Returns
    -------
    ax: matplotlib axes object
    '''

    def slice_with_nans(data, ind):
        '''Insert nans in indices and data where indices non-contiguous'''
        import copy
        import numpy

        ind_nan          = numpy.zeros(len(data))
        ind_nan[:]       = numpy.nan

        # prevent ind from overwrite with deepcopy
        ind_nan[ind]     = copy.deepcopy(ind)
        ind_nan          = ind_nan[ind[0]:ind[-1]]

        # prevent data from overwrite with deepcopy
        data_nan = numpy.copy(data[ind[0]:ind[-1]])
        data_nan[numpy.isnan(ind_nan)] = numpy.nan

        return ind_nan, data_nan

    ax.plot(*slice_with_nans(data, ind), color=color, linewidth=linewidth,
            label=label)

    return ax


def plot_shade_mask(ax, mask):
    '''Shade across x values where boolean mask is `True`'''
    ymin, ymax = ax.get_ylim()
    ax.fill_between(range(len(mask)), ymin, ymax, where=mask,
                    facecolor='gray', alpha=0.5)
    return ax


# Signal
#------------------------------------------------------------------------------

#def plot_cutoff_peak(f_x, S_x, f_x, S_z, peak_idx, idx_f, min_f):
#    '''Plot cuttoff frequencies, axes 0 & 2'''
#
#    b = plt.plot(f, S, 'b', label='')
#    r = plt.plot(f, S, 'r')
#    #legend([b, r], 'HPF acc x axis (surge)', 'HPF acc z axis (heave)')
#
#    plt.plot(f[peak_idx], S[peak_idx], 'o', markersize=10, linewidth=2)
#    plt.plot([f[idx_f], f[idx_f]],[min(S[:]), min_f],'--', linewidth=2)
#    # ['f = '.format(float(round(f[idx_f]*100)/100))],
#
#    return None

def plot_lf_hf(x, xlf, xhf, title=''):
    '''Plot original signal, low-pass filtered, and high-pass filtered signals
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, sharey=True)

    plt.title(title)

    ax1.title.set_text('All freqencies')
    ax1.plot(range(len(x)), x, color=colors[0], linewidth=linewidth,
             label='original')
    ax1.legend(loc='upper right')

    ax2.title.set_text('Low-pass filtered')
    ax2.plot(range(len(xlf)), xlf, color=colors[1], linewidth=linewidth,
             label='low-pass')
    ax2.legend(loc='upper right')

    ax3.title.set_text('High-pass filtered')
    ax3.plot(range(len(xhf)), xhf, color=colors[2], linewidth=linewidth,
             label='high-pass')
    ax3.legend(loc='upper right')

    plt.show()

    return None


def plot_acc_pry_depth(A_g_lf, A_g_hf, pry_deg, depths, glide_mask=None):
    import numpy

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)

    ax1.title.set_text('acceleromter')
    ax1.plot(range(len(A_g_lf)), A_g_lf, color=colors[0])

    ax1.title.set_text('PRH')
    ax2.plot(range(len(pry_deg)), pry_deg, color=colors[1])

    ax3.title.set_text('depth')
    ax3.plot(range(len(depths)), depths, color=colors[2])
    ax3.invert_yaxis()

    if glide_mask is not None:
        ax1 = plot_shade_mask(ax1, glide_mask)
        ax2 = plot_shade_mask(ax2, glide_mask)
        ax3 = plot_shade_mask(ax3, glide_mask)

    plt.show()

    return None


def plot_welch_peaks(f, S, peak_loc=None, title=''):
    '''Plot welch PSD with peaks as scatter points'''
    plt.plot(f, S, linewidth=linewidth)
    plt.title(title)
    plt.xlabel('Fequency (Hz)')
    plt.ylabel('"Power" (g**2 Hz**−1)')

    if peak_loc is not None:
        plt.scatter(f[peak_loc], S[peak_loc], label='peaks')
        plt.legend(loc='upper right')

    plt.show()

    return None


def plot_fft(f, S, dt):
    import numpy

    xf = numpy.linspace(0.0, 1/(2.0*dt), N/2)
    plt.plot(xf, 2.0/N * numpy.abs(S[:N//2]), linewidth=linewidth)
    plt.show()

    return None


def plot_welch_perdiogram(x, fs, nperseg):
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
    ax2.plot(t, data, linewidth=linewidth, label='data')
    ax2.plot(t, data_f, linewidth=linewidth, label='filtered data')
    ax2.set_xlabel('Time [sec]')
    ax2.legend()

    plt.show()

    return None


# ACCELEROMETER AND DIVES
#------------------------------------------------------------------------------

def plot_dives(dv0, dv1, p, dp, t_on, t_off):
    '''Plots depths and delta depths with dive start stop markers'''

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    x0   = t_on[dv0:dv1] - t_on[dv0]
    x1   = t_off[dv0:dv1] - t_on[dv0]

    # Extract start end depths
    y0_p = p[t_on[dv0:dv1]]
    y1_p = p[t_off[dv0:dv1]]

    # Extract start end delta depths
    y0_dp = dp[t_on[dv0:dv1]]
    y1_dp = dp[t_off[dv0:dv1]]

    start = t_on[dv0]
    stop  = t_off[dv1]

    ax1.title.set_text('Dives depths')
    ax1.plot(range(len(p[start:stop])), p[start:stop])
    ax1.scatter(x0, y0_p, label='start')
    ax1.scatter(x1, y1_p, label='stop')
    ax1.set_ylabel('depth (m)')

    ax1.title.set_text('Depth rate of change')
    ax2.plot(range(len(dp[start:stop])), dp[start:stop])
    ax2.scatter(x0, y0_dp, label='start')
    ax2.scatter(x1, y1_dp, label='stop')
    ax2.set_ylabel('depth (dm/t)')
    ax2.set_xlabel('sample')

    for ax in [ax1, ax2]:
        ax.legend(loc='upper right')
        ax.set_xlim([-50, len(dp[start:stop])+50])

    plt.show()

    return None


def plot_dives_pitch(depths, dive_mask, des, asc, pitch, pitch_lf):
    import copy
    import numpy

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

    des_ind = numpy.where(dive_mask & des)[0]
    asc_ind = numpy.where(dive_mask & asc)[0]

    ax1.title.set_text('Dive descents and ascents')
    ax1 = plot_noncontiguous(ax1, depths, des_ind, colors[0], 'descents')
    ax1 = plot_noncontiguous(ax1, depths, asc_ind, colors[1], 'ascents')

    ax1.legend(loc='upper right')
    ax1.invert_yaxis()
    ax1.yaxis.label.set_text('depth (m)')
    ax1.xaxis.label.set_text('samples')


    ax2.title.set_text('Pitch and Low-pass filtered pitch')
    ax2.plot(range(len(pitch)), pitch, color=colors[2], linewidth=linewidth,
            label='pitch')
    ax2.plot(range(len(pitch_lf)), pitch_lf, color=colors[3],
            linewidth=linewidth, label='pitch filtered')
    ax2.legend(loc='upper right')
    ax2.yaxis.label.set_text('Radians')
    ax2.yaxis.label.set_text('Samples')

    plt.show()

    return None


def plot_depth_descent_ascent(depths, dive_mask, des, asc):
    '''Plot depth data for whole deployment, descents, and ascents
    '''
    import numpy

    import utils_dives

    # Indices where depths are descents or ascents
    # TODO if interpolating, can use `des` and `asc` from accelerometry
    des_ind = numpy.where(dive_mask & des)[0]
    asc_ind = numpy.where(dive_mask & asc)[0]

    fig, ax1 = plt.subplots()

    ax1.title.set_text('Dive descents and ascents')
    ax1 = plot_noncontiguous(ax1, depths, des_ind, colors[0], 'descents')
    ax1 = plot_noncontiguous(ax1, depths, asc_ind, colors[1], 'ascents')

    ax1.legend(loc='upper right')
    ax1.invert_yaxis()
    ax1.yaxis.label.set_text('depth (m)')
    ax1.xaxis.label.set_text('samples')

    plt.show()

    return None


def plot_triaxial_depths_speed(data):
    '''Plot triaxial accelerometer data for whole deployment, descents, and
    ascents

    Only x and z axes are ploted since these are associated with stroking
    '''
    import numpy

    # TODO return to multiple inputs rather than dataframe

    fig, axes = plt.subplots(3, 3, sharex='col', sharey='row')
    ((ax1, ax4, ax7), (ax2, ax5, ax8), (ax3, ax6, ax9)) = axes

    # Create mask of all True for length of depths
    all_ind = numpy.arange(0, len(data), dtype=int)

    cols = [('x', data['Ax_g'], [ax1, ax2, ax3]),
            ('y', data['Ay_g'], [ax4, ax5, ax6]),
            ('z', data['Az_g'], [ax7, ax8, ax9])]

    for label, y, axes in cols:
        axes[0].title.set_text('Accelerometer {}-axis'.format(label))
        axes[0].plot(range(len(y)), y, color=colors[0],
                     linewidth=linewidth, label='x')

        axes[1].title.set_text('Depths')
        axes[1] = plot_noncontiguous(axes[1], data['depth'], all_ind, color=colors[1])
        axes[1].invert_yaxis()

        axes[2] = plot_noncontiguous(axes[2], data['propeller'], all_ind,
                color=colors[2], label='propeller')

    plt.show()

    return None


def plot_triaxial_descent_ascent(Ax, Az, des, asc):
    '''Plot triaxial accelerometer data for whole deployment, descents, and
    ascents

    Only x and z axes are ploted since these are associated with stroking
    '''
    import numpy

    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

    # Convert boolean mask to indices
    des_ind = numpy.where(des)[0]
    asc_ind = numpy.where(asc)[0]

    cols = [('x', Ax, [ax1, ax2]),
            ('z', Az, [ax3, ax4])]

    for label, data, axes in cols:
        axes[0].title.set_text('Whole {}'.format(label))
        axes[0].plot(range(len(data)), data, color=colors[0],
                     linewidth=linewidth, label='{}'.format(label))

        axes[1].title.set_text('Descents & Ascents {}'.format(label))
        axes[1] = plot_noncontiguous(axes[1], data, des_ind, color=colors[1],
                                     label='descents')
        axes[1] = plot_noncontiguous(axes[1], data, asc_ind, color=colors[2],
                                     label='ascents')
        axes[1].legend(loc='upper right')

    plt.show()

    return None


# Pitch, Roll, Heading
#------------------------------------------------------------------------------

def plot_prh_des_asc(p, r, h, asc, des):
    import matplotlib.pyplot as plt
    import numpy

    # Convert boolean mask to indices
    des_ind = numpy.where(des)[0]
    asc_ind = numpy.where(asc)[0]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col')

    ax1.title.set_text('Pitch')
    ax1 = plot_noncontiguous(ax1, p, des_ind, colors[0], 'descents')
    ax1 = plot_noncontiguous(ax1, p, asc_ind, colors[1], 'ascents')

    ax1.title.set_text('Roll')
    ax2 = plot_noncontiguous(ax2, r, des_ind, colors[0], 'descents')
    ax2 = plot_noncontiguous(ax2, r, asc_ind, colors[1], 'ascents')

    ax1.title.set_text('Heading')
    ax3 = plot_noncontiguous(ax3, h, des_ind, colors[0], 'descents')
    ax3 = plot_noncontiguous(ax3, h, asc_ind, colors[1], 'ascents')

    for ax in [ax1, ax2, ax3]:
        ax.legend(loc="upper right")

    plt.ylabel('Radians')
    plt.xlabel('Samples')

    plt.show()

    return None


def plot_prh_filtered(p, r, h, p_lf, r_lf, h_lf):
    import numpy

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col')

    #rad2deg = lambda x: x*180/numpy.pi

    ax1.title.set_text('Pitch')
    ax1.plot(range(len(p)), p, color=colors[0], linewidth=linewidth,
            label='original')
    ax1.plot(range(len(p_lf)), p_lf, color=colors[1], linewidth=linewidth,
            label='filtered')

    ax2.title.set_text('Roll')
    ax2.plot(range(len(r)), r, color=colors[2], linewidth=linewidth,
            label='original')
    ax2.plot(range(len(r_lf)), r_lf, color=colors[3], linewidth=linewidth,
            label='filtered')

    ax3.title.set_text('Heading')
    ax3.plot(range(len(h)), h, color=colors[4], linewidth=linewidth,
            label='original')
    ax3.plot(range(len(h_lf)), h_lf, color=colors[5], linewidth=linewidth,
            label='filtered')

    plt.ylabel('Radians')
    plt.xlabel('Samples')
    for ax in [ax1, ax2, ax3]:
        ax.legend(loc="upper right")

    plt.show()

    return None


def plot_swim_speed(exp_ind, swim_speed):
    import numpy

    fig, ax = plt.subplots()

    ax.title.set_text('Swim speed from depth change and pitch angle (m/s^2')
    ax.plot(exp_ind, swim_speed, linewidth=linewidth, label='speed')
    ymax = numpy.ceil(swim_speed[~numpy.isnan(swim_speed)].max())
    ax.set_ylim(0, ymax)
    ax.legend(loc='upper right')

    plt.show()

    return ax


def plot_get_fluke(depths, pitch, A_g_hf, dive_ind, nn, fs):
    import matplotlib.pyplot as plt

    import utils

    # Get max acc from 0:nn over axes [0,2] (x, z)
    maxy = numpy.max(numpy.max(Ahf[round(dive_ind[0, 0] * fs): round(dive_ind[nn, 1] * fs), [0,2]]))

    # Selection plot of depths an HPF x & z axes
    ax1 = plt.gca()
    ax2 = ax1.twiny()

    ax1.plot(range(len(depths)), depths, label='depths')
    ax2.plot(range(len(pitch)), Ahf[:, 0], lable='HPF x')
    ax2.plot(range(len(pitch)), Ahf[:, 2], lable='HPF z')

    ax1.invert_yaxis()

    for ax in [ax1, ax2]:
        ax.set_ylim(-2*maxy, 2*maxy)
        #ax.ytick(round(-2*maxy*10) / 10: 0.1: 2 * maxy)

    plt.legend(loc='upper right')
    plt.title('Zoom in to find thresholds for fluking, then enter it for J')

    plt.show()

    return None


# Glides
#------------------------------------------------------------------------------

def plot_glide_depths(depths, data_sgl_mask):
    '''Plot depth at glides'''
    import numpy

    fig, ax = plt.subplots()

    ax = plot_noncontiguous(ax, depths, numpy.where(data_sgl_mask)[0])
    ax.invert_yaxis()

    plt.show()

    return None


def plot_sgls(depths, data_sgl_mask, sgls, sgl_mask, pitch_lf, roll_lf, heading_lf):

    import matplotlib.pyplot as plt
    import numpy

    from bodycondition import utils
    from bodycondition import utils_glides

    sgl_ind    = numpy.where(data_sgl_mask)[0]
    notsgl_ind = numpy.where(~data_sgl_mask)[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot glides
    ax1 = plot_noncontiguous(ax1, depths, sgl_ind, colors[0], 'glides')
    ax1 = plot_noncontiguous(ax1, depths, notsgl_ind, colors[1], 'not glides')

    ax1.invert_yaxis()
    ax1.yaxis.label.set_text('depth (m)')
    ax1.xaxis.label.set_text('samples')
    ax1.legend(loc='upper right')

    # Plot PRH
    ax2.plot(range(len(depths)), numpy.rad2deg(pitch_lf), color=colors[2], label='pitch')
    ax2.plot(range(len(depths)), numpy.rad2deg(roll_lf), color=colors[3], label='roll')
    ax2.plot(range(len(depths)), numpy.rad2deg(heading_lf), color=colors[4],
                                               label='heading')
    ax2.yaxis.label.set_text('degrees')
    ax2.xaxis.label.set_text('samples')
    ax2.legend(loc='upper right')

    # Get dives within mask
    gg = sgls[sgl_mask]

    # Get midpoint of dive occurance
    x = (gg['start_idx'] + (gg['stop_idx'] - gg['start_idx'])/2)
    x = x.values.astype(float)

    # Get depthh at midpoint
    y = depths[numpy.round(x).astype(int)]

    # For each dive_id, sgl_id pair, create annotation string, apply
    dids = gg['dive_id'].values.astype(int)
    sids = list(gg.index)
    n = ['d:{} s:{}'.format(did, sid) for did, sid in zip(dids, sids)]

    # Draw annotations
    for i, txt in enumerate(n):
        ax1.annotate(txt, (x[i],y[i]))

    # Plot shaded areas
    ax1 = plot_shade_mask(ax1, ~data_sgl_mask)
    ax2 = plot_shade_mask(ax2, ~data_sgl_mask)

    plt.show()

    return None


def sgl_density(sgls, max_depth=20, textstr='', fname=False):
    '''Plot density of subglides over time for whole exp, des, and asc'''
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy


    sns.set(style="white", color_codes=True)

    fig = plt.figure()

    # time, mid between start and finish
    sgl_x = sgls['start_idx'] + ((sgls['stop_idx']-sgls['start_idx'])/2)

    # depth, calc avg over sgl time
    sgl_y = sgls['mean_depth']

    g = sns.jointplot(x=sgl_x, y=sgl_y, kind='hex', stat_func=None)

    g.fig.axes[0].set_ylim(0, max_depth)
    g.fig.axes[0].invert_yaxis()
    g.set_axis_labels(xlabel='Time', ylabel='Depth (m)')

    # Add text annotation top left if `textstr` passed
    if textstr:
        props = dict(boxstyle='round', facecolor='grey', alpha=0.1)
        g.fig.axes[0].text(0.05, 0.20, textstr, transform=g.fig.axes[0].transAxes,
                           fontsize=14, verticalalignment='top', bbox=props)

    if fname:
        g.savefig(filename=fname)#, dpi=300)
    else:
        plt.show()

    return None
