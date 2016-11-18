def plot_dives(dv0, dv1, p, dp, t_on, t_off):
    '''Plots depths and delta depths with dive start stop markers'''
    import matplotlib.pyplot as plt

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.title.set_text('Dives')

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
    ax1.plot(range(len(p[start:stop])), p[start:stop])
    ax1.scatter(x0, y0_p, color='red', label='start')
    ax1.scatter(x1, y1_p, color='blue', label='stop')
    ax1.set_ylabel('depth (m)')

    ax2.plot(range(len(dp[start:stop])), dp[start:stop])
    ax2.scatter(x0, y0_dp, color='red', label='start')
    ax2.scatter(x1, y1_dp, color='blue', label='stop')
    ax2.set_ylabel('depth (dm/t)')
    ax2.set_xlabel('sample')

    for ax in [ax1, ax2]:
        ax.legend(loc='upper right')
        ax.set_xlim([-50, len(dp[start:stop])+50])

    plt.show()

    return None


def plot_noncontiguous(ax, data, ind, label=''):
    '''Plot non-contiguous slice of data

    Args
    ----
    data: 1-D numpy array
    ind: indices of data to be plotted

    Returns
    -------
    ax: matplotlib axes object
    '''
    import matplotlib.pyplot as plt

    def slice_with_nans(data, ind):
        '''Insert nans in indices and data where indices non-contiguous'''
        import numpy

        ind_nan          = numpy.zeros(len(data))
        ind_nan[:]       = numpy.nan

        # prevent ind from overwrite with deepcopy
        ind_nan[ind]     = numpy.copy(ind)
        ind_nan          = ind_nan[ind[0]:ind[-1]]

        # prevent data from overwrite with deepcopy
        data_nan = numpy.copy(data[ind[0]:ind[-1]])
        data_nan[numpy.isnan(ind_nan)] = numpy.nan

        return ind_nan, data_nan

    ax.plot(*slice_with_nans(data, ind), label=label)

    return ax


def plot_welch(f, S):
    import matplotlib.pyplot as plt

    plt.plot(f,S)
    plt.show()

    return None


def plot_fft(f, S, dt):
    import matplotlib.pyplot as plt
    import numpy

    xf = numpy.linspace(0.0, 1/(2.0*dt), N/2)
    plt.plot(xf, 2.0/N * numpy.abs(S[:N//2]))
    plt.show()

    return None


def plot_welch_perdiogram(x, fs, nperseg):
    import matplotlib.pyplot as plt
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


def plot_descent_ascent(A, DES, ASC):#, fs_a
    '''Plot accel. for whole deployment, descents, and ascents'''
    import matplotlib.pyplot as plt

    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2)

    #ax1 = plot_power_spec(ax1, 
    #ax2 = plot_power_spec(ax2, A[DES, :], fs_a, title='Descent')
    #ax3 = plot_power_spec(ax3, A[ASC, :], fs_a, title='Ascent')

    ax1.plot(range(len(A[:,0])), A[:,0], label='x')
    ax1.title.set_text('Whole x')

    ax4.plot(range(len(A[:,2])), A[:,2], label='z')
    ax4.title.set_text('Whole z')

    xlim_whole = (*(ax1.get_xlim()), *(ax4.get_xlim()))
    ylim_whole = (*(ax1.get_ylim()), *(ax4.get_ylim()))

    for a in [ax1, ax2]:
        a.set_xlim(min(xlim_whole), max(xlim_whole))
        a.set_ylim(min(ylim_whole), max(ylim_whole))

    ax2 = plot_noncontiguous(ax2, A[:,0], DES, label='x')
    ax2.title.set_text('Descent x')

    ax5 = plot_noncontiguous(ax5, A[:,2], DES, label='z')
    ax5.title.set_text('Descent z')

    ax3 = plot_noncontiguous(ax3, A[:,0], ASC, label='x')
    ax3.title.set_text('Ascent x')

    ax6 = plot_noncontiguous(ax6, A[:,2], ASC, label='z')
    ax6.title.set_text('Ascent z')

    xlim_desasc = (*(ax2.get_xlim()), *(ax5.get_xlim()),
                   *(ax3.get_xlim()), *(ax6.get_xlim()))
    ylim_desasc = (*(ax2.get_ylim()), *(ax5.get_ylim()),
                   *(ax3.get_ylim()), *(ax6.get_ylim()))

    for a in [ax2, ax5, ax3, ax6]:
        a.set_xlim(min(xlim_desasc), max(xlim_desasc))
        a.set_ylim(min(ylim_desasc), max(ylim_desasc))

    plt.show()

    return None


def plot_pitch_roll(pitch, roll, pitch_lf, roll_lf):
    import matplotlib.pyplot as plt
    import numpy

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')

    rad2deg = lambda x: x*180/numpy.pi

    ax1.plot(range(len(pitch)), rad2deg(pitch), label='original')
    ax1.plot(range(len(pitch_lf)), rad2deg(pitch_lf), label='filtered')
    ax1.title.set_text('Pitch')

    ax2.plot(range(len(roll)), rad2deg(roll), label='original')
    ax2.plot(range(len(roll_lf)), rad2deg(roll_lf), label='filtered')
    ax2.title.set_text('Roll')

    plt.ylabel('Degrees')
    plt.xlabel('Samples')
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    plt.show()

    return None


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


