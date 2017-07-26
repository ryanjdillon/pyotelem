
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

    # Indices where depths are descents or ascents
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


