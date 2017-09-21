'''
Plotting functions for dynamic movement of the animal, including pitch, roll,
and heading as well as speed.
'''

import matplotlib.pyplot as plt

from . import plotconfig as _plotconfig
from .plotconfig import _colors, _linewidth

def plot_prh_des_asc(p, r, h, asc, des):
    '''Plot pitch, roll, and heading during the descent and ascent dive phases

    Args
    ----
    p: ndarray
        Derived pitch data
    r: ndarray
        Derived roll data
    h: ndarray
        Derived heading data
    des: ndarray
        boolean mask for slicing descent phases of dives from tag dta
    asc: ndarray
        boolean mask for slicing asccent phases of dives from tag dta
    '''
    import matplotlib.pyplot as plt
    import numpy

    from . import plotutils

    # Convert boolean mask to indices
    des_ind = numpy.where(des)[0]
    asc_ind = numpy.where(asc)[0]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col')

    ax1.title.set_text('Pitch')
    ax1 = plotutils.plot_noncontiguous(ax1, p, des_ind, _colors[0], 'descents')
    ax1 = plotutils.plot_noncontiguous(ax1, p, asc_ind, _colors[1], 'ascents')

    ax1.title.set_text('Roll')
    ax2 = plotutils.plot_noncontiguous(ax2, r, des_ind, _colors[0], 'descents')
    ax2 = plotutils.plot_noncontiguous(ax2, r, asc_ind, _colors[1], 'ascents')

    ax1.title.set_text('Heading')
    ax3 = plotutils.plot_noncontiguous(ax3, h, des_ind, _colors[0], 'descents')
    ax3 = plotutils.plot_noncontiguous(ax3, h, asc_ind, _colors[1], 'ascents')

    for ax in [ax1, ax2, ax3]:
        ax.legend(loc="upper right")

    plt.ylabel('Radians')
    plt.xlabel('Samples')

    plt.show()

    return None


def plot_prh_filtered(p, r, h, p_lf, r_lf, h_lf):
    '''Plot original and low-pass filtered PRH data

    Args
    ----
    p: ndarray
        Derived pitch data
    r: ndarray
        Derived roll data
    h: ndarray
        Derived heading data
    p_lf: ndarray
        Low-pass filtered pitch data
    r_lf: ndarray
        Low-pass filtered roll data
    h_lf: ndarray
        Low-pass filtered heading data
    '''
    import numpy

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col')

    #rad2deg = lambda x: x*180/numpy.pi

    ax1.title.set_text('Pitch')
    ax1.plot(range(len(p)), p, color=_colors[0], linewidth=_linewidth,
            label='original')
    ax1.plot(range(len(p_lf)), p_lf, color=_colors[1], linewidth=_linewidth,
            label='filtered')

    ax2.title.set_text('Roll')
    ax2.plot(range(len(r)), r, color=_colors[2], linewidth=_linewidth,
            label='original')
    ax2.plot(range(len(r_lf)), r_lf, color=_colors[3], linewidth=_linewidth,
            label='filtered')

    ax3.title.set_text('Heading')
    ax3.plot(range(len(h)), h, color=_colors[4], linewidth=_linewidth,
            label='original')
    ax3.plot(range(len(h_lf)), h_lf, color=_colors[5], linewidth=_linewidth,
            label='filtered')

    plt.ylabel('Radians')
    plt.xlabel('Samples')
    for ax in [ax1, ax2, ax3]:
        ax.legend(loc="upper right")

    plt.show()

    return None


def plot_swim_speed(exp_ind, swim_speed):
    '''Plot the swim speed during experimental indices

    Args
    ----
    exp_ind: ndarray
        Indices of tag data where experiment is active
    swim_speed: ndarray
        Swim speed data at sensor sampling rate
    '''
    import numpy

    fig, ax = plt.subplots()

    ax.title.set_text('Swim speed from depth change and pitch angle (m/s^2')
    ax.plot(exp_ind, swim_speed, linewidth=_linewidth, label='speed')
    ymax = numpy.ceil(swim_speed[~numpy.isnan(swim_speed)].max())
    ax.set_ylim(0, ymax)
    ax.legend(loc='upper right')

    plt.show()

    return ax
