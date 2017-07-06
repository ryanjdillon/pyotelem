
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


