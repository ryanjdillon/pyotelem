# TODO REMOVE

# 3.2.2_Determine `stroke_f` fluking rate and cut-off frequency
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_power_spec(ax, acc_ax, fs_a, title=''):
    S, f = speclev(acc_ax, 512, fs_a)

    # TODO check that S.size[1] == 3
    for nn in range(3):
        # make 0.1 smaller if it is not finding the peak you expect
        peakloc, peakmag = peakfinder(S[: , nn], 0.1)
        peakloc[0] = []
        peakmag[0] = []
        smooth_S    = runmean(S[: , nn], 10)
        peakmag    = peakmag - smooth_S[peakloc]
        _, peak    = max(peakmag)
        peak       = peakloc[peak]

        min_f, idx_f = min(S[:peak, nn])
        cutoff_low[stroke_f_n]   = f[idx_f]

        plot(f[peak], S[peak, nn], 'go', 'markersize', 10, 'linewidth', 2)
        plot([f[idx_f], f[idx_f]],[min(S[:,nn]), min_f],'k--','linewidth',2)

        text(f[idx_f], min(min(S[:,[1, 3]])),
             ['f = '.format(float(round(f[idx_f]*100)/100))],
             'horizontalalignment','right','color',
             cs[nn],'verticalalignment',va[nn])

        stroke_f[stroke_f_n]     = f[peak]
        stroke_f_mag[stroke_f_n] = S[peak,nn]

        stroke_f_n = stroke_f_n+1

    _, b = max(stroke_f_mag[1: 2])

    if b == 2:
        v1 = 'top'
        v2 = 'bottom'
    else:
        v1 = 'bottom'
        v2 = 'top'

    text(stroke_f[1],
         stroke_f_mag[0] + 2 * sign( len(v1) - len(v2)),
         num2str( round( stroke_f[0] * 100) / 100),
         'verticalalignment', v1, 'horizontalalignment', 'center')

    text(stroke_f[2],
         stroke_f_mag[1] - 2 * sign( len(v1) - len(v2)),
         num2str( round( stroke_f[1] * 100) / 100),
         'verticalalignment', v2, 'horizontalalignment', 'center')

    b = plot(f, S[:, 0], 'b')
    r = plot(f, S[:, 2], 'r')

    ax1 = gca

    set(get(ax1, 'Xlabel'), 'String', [{'$\bf\ Frequency \hspace{1mm} (Hz) $'}],
        'interpreter', 'latex', 'FontSize', 8, 'FontName', 'Arial')

    ys = get(ax1, 'ylim')

    # to write the name of the pannel
    hp1 = text( 0.02, diff(ys) * .92 + min(ys), title, 'FontSize', 12,
            'FontWeight', 'bold', 'horizontalalignment', 'left')

    legend([b, r], 'HPF acc x axis (surge)', 'HPF acc z axis (heave)')

    return ax



# 7.2 using the body rotations (pry) estimated using the
# magnetometer method (use n=1)
#----------------------------------------------------------------------

def get_J_pry_mag():
    alpha = 25
    cutoff_low_mag = cutoff_low

    MagAcc, pry, Sa, GL, KK = magnet_rot_sa(Aw, Mw, fs_a, stroke_f, f, alpha, 1,
                                            k, J, tmax)

    # TODO PLOT: find fluke thresholds
    figure(6)
    subplot1 = subplot(1, 4, 1: 3)

    ax1 = plotyy(range(len(depths)), depths, range(len(pry[: , 0])), pry[: , 0])
    set(ax1(1), 'ydir', 'rev', 'ylim', [0, max(depths)])

    maxy = max(pry(round(T[0, 0] * fs): round(T[nn, 1] * fs)))

    set(ax1(2),
        'ylim', [-2 * maxy 2 * maxy],
        'ytick', round(-2 * maxy * 10) / 10: 0.1: 2 * maxy)

    set(ax1(2), 'nextplot', 'add')
    legend('Depth', 'rotations in y axis')
    title('Zoom in to find appropriate thresholds for fluking, then enter it for J')



    # body rotations ascent in radians
    flukePA = pry[ASC, 0]
    flukePD = pry[DES, 0]

    TOTAL = abs([flukePA flukePD])
    Y = buffer(TOTAL, 2 * round(1 / stroke_f * fs_a))

    # TODO PLOT: find fluke thresholds
    subplot2 = subplot(1, 4, 4)

    hist(max(Y), 100)
    set(subplot2, 'xlim', [0 max(max(Y))])

    # Choose a value for J based on the histogram for:
    #   pry(:,1), then when detecting glides in the next step use magnet_rot_sa
    #   function with n=1

    J =  # in radians
    tmax = 1 / stroke_f  # in seconds

    MagAcc, pry, Sa, GL, KK = magnet_rot_sa(Aw, Mw, fs_a, stroke_f, f, alpha, 1,
                                            k, J,tmax)







# TODO plots

plt.plot(k, p_dive, 'g')

plt.plot(k, p_nodive, 'r', 'linewidth', 2)
set(gca, 'ydir', 'rev', 'ylim', [min(depths) max(depths)])
legend('Dive', 'Not Dive', 'location', 'southeast')

plt.plot(p_asc, 'k', 'linewidth', 2)

plt.plot(p_des, 'b', 'linewidth', 2)


ax2 = plt.plotyy(k, SwimSp, k, pry[:, 1])
plt.ylim(0, max(SwimSp)])
set(ax2(2), 'nextplot', 'add')


plt.plot(KK[: , 1]*fs, 0, 'r*', label='body rotations')
plt.ylim(min(pry([ASC, DES]]), max(pry([ASC, DES])))
plt.legend(loc='upper right')

# glides detected with the body rotations (pry)
plt.plot(t * fs, p_gl, 'm:', linewidth=3, label='glides')
plt.legend(loc='upper right')


# glides detected with the hpf acc
plt.plot(t * fs, p_gla, 'y:', linewidth=3, label='glides')
plt.legend(loc='upper right')

