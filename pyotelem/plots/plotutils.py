import matplotlib.pyplot as plt


def roundup(x, order):
    return x if x % 10**order == 0 else x + 10**order - x % 10**order


def magnitude(x):
    import math
    return int(math.floor(math.log10(x)))


def hourminsec(n_seconds):
    '''Generate a string of hours and minutes from total number of seconds'''

    hours, remainder = divmod(n_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return abs(hours), abs(minutes), abs(seconds)


def nsamples_to_hourminsec(x, pos):
    '''Convert axes labels to experiment duration in hours/minutes/seconds

    Matplotlib FuncFormatter function
    https://matplotlib.org/examples/pylab_examples/custom_ticker1.html
    '''

    h, m, s = hourminsec(x/16.0)
    return '{:.0f}h {:2.0f}′ {:2.1f}″'.format(h, m, s)


def nsamples_to_hourmin(x, pos):
    '''Convert axes labels to experiment duration in hours/minutes

    Matplotlib FuncFormatter function
    https://matplotlib.org/examples/pylab_examples/custom_ticker1.html
    '''

    h, m, s = hourminsec(x/16.0)
    return '{:.0f}h {:2.0f}′'.format(h, m+round(s))


def nsamples_to_minsec(x, pos):
    '''Convert axes labels to experiment duration in minutes/seconds

    Matplotlib FuncFormatter function
    https://matplotlib.org/examples/pylab_examples/custom_ticker1.html
    '''
    h, m, s = hourminsec(x/16.0)
    return '{:2.0f}′ {:2.1f}″'.format(m+(h*60), s)


def add_alpha_labels(axes, xpos=0.03, ypos=0.95, suffix='', color=None,
        fontsize=14, fontweight='normal', boxstyle='square', facecolor='white',
        edgecolor='white', alpha=1.0):
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
                      fontweight=fontweight,)

        bbox = dict(boxstyle=boxstyle,
                     facecolor=facecolor,
                     edgecolor=edgecolor,
                     alpha=alpha)

        ax.text(xpos[i], ypos[i], '{}{}'.format(label, suffix),
                transform=ax.transAxes, fontsize=fontsize,
                verticalalignment='top', bbox=bbox, **kwargs)
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


def plot_noncontiguous(ax, data, ind, color='black', label='', offset=0,
        linewidth=0.5, linestyle='-'):
    '''Plot non-contiguous slice of data

    Args
    ----
    data: 1-D numpy array
    ind: indices of data to be plotted

    Returns
    -------
    ax: matplotlib axes object
    '''

    def slice_with_nans(ind, data, offset):
        '''Insert nans in indices and data where indices non-contiguous'''
        import copy
        import numpy

        ind_nan          = numpy.zeros(len(data))
        ind_nan[:]       = numpy.nan

        # prevent ind from overwrite with deepcopy
        ind_nan[ind-offset] = copy.deepcopy(ind)
        #ind_nan             = ind_nan[ind[0]-offset:ind[-1]-offset]

        # prevent data from overwrite with deepcopy
        data_nan = copy.deepcopy(data)
        data_nan[numpy.isnan(ind_nan)] = numpy.nan

        return ind_nan, data_nan

    x, y = slice_with_nans(ind, data, offset)
    ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle,
            label=label)

    return ax


def plot_shade_mask(ax, ind, mask, facecolor='gray'):
    '''Shade across x values where boolean mask is `True`'''
    ymin, ymax = ax.get_ylim()
    ax.fill_between(ind, ymin, ymax, where=mask,
                    facecolor=facecolor, alpha=0.5)
    return ax
