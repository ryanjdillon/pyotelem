import matplotlib.pyplot as plt

from .plotconfig import _colors, _linewidth

def add_alpha_labels(axes, xpos=0.03, ypos=0.95, color=None, boxstyle='square',
        facecolor='white', edgecolor='white', alpha=1.0):
    '''Add sequential alphbet labels to subplot axes

    e.g. A., B., C., etc.
    '''
    import seaborn
    import string
    import numpy

    if not numpy.iterable(xpos):
        xpos = [xpos,]*len(axes)
        ypos = [ypos,]*len(axes)

    if (len(xpos) > 1) or (len(ypos) > 1):
        try:
            assert (len(axes) == len(xpos))
        except AssertionError as e:
            e.args += 'xpos iterable must be same length as axes'
            raise
        try:
            assert (len(axes) == len(ypos))
        except AssertionError as e:
            e.args += 'ypos iterable must be same length as axes'
            raise
    else:
        xpos = [xpos,]
        ypos = [ypos,]

    colors = seaborn.color_palette()
    abc = string.ascii_uppercase

    for i, (label, ax) in enumerate(zip(abc[:len(axes)], axes)):
        if color is None:
            color = colors[i]
        kwargs = dict(color=color,
                      fontweight='bold',)

        bbox = dict(boxstyle=boxstyle,
                     facecolor=facecolor,
                     edgecolor=edgecolor,
                     alpha=1.0)

        ax.text(xpos[i], ypos[i], '{}.'.format(label), transform=ax.transAxes,
                fontsize=14, verticalalignment='top', bbox=bbox, **kwargs)
    return axes


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


def plot_noncontiguous(ax, data, ind, color=_colors[0], label=''):
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

    ax.plot(*slice_with_nans(data, ind), color=color, linewidth=_linewidth,
            label=label)

    return ax


def plot_shade_mask(ax, mask):
    '''Shade across x values where boolean mask is `True`'''
    ymin, ymax = ax.get_ylim()
    ax.fill_between(range(len(mask)), ymin, ymax, where=mask,
                    facecolor='gray', alpha=0.5)
    return ax
