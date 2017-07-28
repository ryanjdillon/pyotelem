import matplotlib.pyplot as plt

from . import plotconfig as _plotconfig
from .plotconfig import _colors

def plot_glide_depths(depths, data_sgl_mask):
    '''Plot depth at glides'''
    import numpy

    from . import plotutils

    fig, ax = plt.subplots()

    ax = plotutils.plot_noncontiguous(ax, depths, numpy.where(data_sgl_mask)[0])
    ax.invert_yaxis()

    plt.show()

    return None


def plot_sgls(depths, data_sgl_mask, sgls, sgl_mask, pitch_lf, roll_lf, heading_lf):

    import matplotlib.pyplot as plt
    import numpy

    from . import plotutils

    sgl_ind    = numpy.where(data_sgl_mask)[0]
    notsgl_ind = numpy.where(~data_sgl_mask)[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot glides
    ax1 = plotutils.plot_noncontiguous(ax1, depths, sgl_ind, _colors[0],
                                      'glides')
    ax1 = plotutils.plot_noncontiguous(ax1, depths, notsgl_ind, _colors[1],
                                       'not glides')

    ax1.invert_yaxis()
    ax1.yaxis.label.set_text('depth (m)')
    ax1.xaxis.label.set_text('samples')
    ax1.legend(loc='upper right')

    # Plot PRH
    ax2.plot(range(len(depths)), numpy.rad2deg(pitch_lf), color=_colors[2], label='pitch')
    ax2.plot(range(len(depths)), numpy.rad2deg(roll_lf), color=_colors[3], label='roll')
    ax2.plot(range(len(depths)), numpy.rad2deg(heading_lf), color=_colors[4],
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

    # TODO add A/ B, bottom left
    # TODO move textbox bottom right
    # TODO set limits for density the same
    # TODO Update xaxis, units time elapsed
    # TODO save as svg

    # Make jointplots as subplots
    # http://stackoverflow.com/a/35044845/943773

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

    ## TODO add colorbar
    ## http://stackoverflow.com/a/29909033/943773
    #cax = g.fig.add_axes([1, 0.35, 0.01, 0.2])
    #plt.colorbar(cax=cax)

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
