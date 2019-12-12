"""
Utility functions for plotting
"""


def roundup(x, order):
    """Round a number to the passed order

    Args
    ----
    x: float
        Number to be rounded
    order: int
        Order to which `x` should be rounded

    Returns
    -------
    x_round: float
        The passed value rounded to the passed order
    """
    return x if x % 10 ** order == 0 else x + 10 ** order - x % 10 ** order


def magnitude(x):
    """Determine the magnitude of a number

    Args
    ----
    x: float
        Number whose magnitude to find

    Returns
    -------
    mag: int
        Magnitude of passed number
    """
    import math

    return int(math.floor(math.log10(x)))


def hourminsec(n_seconds):
    """Generate a string of hours and minutes from total number of seconds

    Args
    ----
    n_seconds: int
        Total number of seconds to calculate hours, minutes, and seconds from

    Returns
    -------
    hours: int
        Number of hours in `n_seconds`
    minutes: int
        Remaining minutes in `n_seconds` after number of hours
    seconds: int
        Remaining seconds in `n_seconds` after number of minutes
    """

    hours, remainder = divmod(n_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return abs(hours), abs(minutes), abs(seconds)


def nsamples_to_hourminsec(x, pos):
    """Convert axes labels to experiment duration in hours/minutes/seconds

    Notes
    -----
    Matplotlib FuncFormatter function
    https://matplotlib.org/examples/pylab_examples/custom_ticker1.html
    """

    h, m, s = hourminsec(x / 16.0)
    return "{:.0f}h {:2.0f}′ {:2.1f}″".format(h, m, s)


def nsamples_to_hourmin(x, pos):
    """Convert axes labels to experiment duration in hours/minutes

    Notes
    -----
    Matplotlib FuncFormatter function
    https://matplotlib.org/examples/pylab_examples/custom_ticker1.html
    """

    h, m, s = hourminsec(x / 16.0)
    return "{:.0f}h {:2.0f}′".format(h, m + round(s))


def nsamples_to_minsec(x, pos):
    """Convert axes labels to experiment duration in minutes/seconds

    Notes
    -----
    Matplotlib FuncFormatter function
    https://matplotlib.org/examples/pylab_examples/custom_ticker1.html
    """
    h, m, s = hourminsec(x / 16.0)
    return "{:2.0f}′ {:2.1f}″".format(m + (h * 60), s)


def add_alpha_labels(
    axes,
    xpos=0.03,
    ypos=0.95,
    suffix="",
    color=None,
    fontsize=14,
    fontweight="normal",
    boxstyle="square",
    facecolor="white",
    edgecolor="white",
    alpha=1.0,
):
    """Add sequential alphbet labels to subplot axes

    Args
    ----
    axes: list of pyplot.ax
        A list of matplotlib axes to add the label labels to
    xpos: float or array_like
        X position(s) of labels in figure coordinates
    ypos: float or array_like
        Y position(s) of labels in figure coordinates
    suffix: str
        String to append to labels (e.g. '.' or ' name)
    color: matplotlib color
        Color of labels
    fontsize: int
        Alppa fontsize
    fontweight: matplotlib fontweight
        Alpha fontweight
    boxstyle: matplotlib boxstyle
        Alpha boxstyle
    facecolor: matplotlib facecolor
        Color of box containing label
    edgecolor: matplotlib edgecolor
        Color of box'es border containing label
    alpha: float
        Transparency of label

    Returns
    -------
    axes: list of pyplot.ax
        A list of matplotlib axes objects with alpha labels added
    """
    import seaborn
    import string
    import numpy

    if not numpy.iterable(xpos):
        xpos = [xpos] * len(axes)
        ypos = [ypos] * len(axes)

    if (len(xpos) > 1) or (len(ypos) > 1):
        try:
            assert len(axes) == len(xpos)
        except AssertionError as e:
            e.args += "xpos iterable must be same length as axes"
            raise
        try:
            assert len(axes) == len(ypos)
        except AssertionError as e:
            e.args += "ypos iterable must be same length as axes"
            raise
    else:
        xpos = [xpos]
        ypos = [ypos]

    colors = seaborn.color_palette()
    abc = string.ascii_uppercase

    for i, (label, ax) in enumerate(zip(abc[: len(axes)], axes)):
        if color is None:
            color = colors[i]
        kwargs = dict(color=color, fontweight=fontweight)

        bbox = dict(
            boxstyle=boxstyle, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha
        )

        ax.text(
            xpos[i],
            ypos[i],
            "{}{}".format(label, suffix),
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="top",
            bbox=bbox,
            **kwargs
        )
    return axes


def merge_limits(axes, xlim=True, ylim=True):
    """Set maximum and minimum limits from list of axis objects to each axis

    Args
    ----
    axes: iterable
        list of `matplotlib.pyplot` axis objects whose limits should be modified
    xlim: bool
        Flag to set modification of x axis limits
    ylim: bool
        Flag to set modification of y axis limits
    """

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


def plot_noncontiguous(
    ax, data, ind, color="black", label="", offset=0, linewidth=0.5, linestyle="-"
):
    """Plot non-contiguous slice of data

    Args
    ----
    data: ndarray
        The data with non continguous regions to plot
    ind: ndarray
        indices of data to be plotted
    color: matplotlib color
        Color of plotted line
    label: str
        Name to be shown in legend
    offset: int
        The number of index positions to reset start of data to zero
    linewidth: float
        The width of the plotted line
    linstyle: str
        The char representation of the plotting style for the line

    Returns
    -------
    ax: pyplot.ax
        Axes object with line glyph added for non-contiguous regions
    """

    def slice_with_nans(ind, data, offset):
        """Insert nans in indices and data where indices non-contiguous"""
        import copy
        import numpy

        ind_nan = numpy.zeros(len(data))
        ind_nan[:] = numpy.nan

        # prevent ind from overwrite with deepcopy
        ind_nan[ind - offset] = copy.deepcopy(ind)
        # ind_nan = ind_nan[ind[0]-offset:ind[-1]-offset]

        # prevent data from overwrite with deepcopy
        data_nan = copy.deepcopy(data)
        data_nan[numpy.isnan(ind_nan)] = numpy.nan

        return ind_nan, data_nan

    x, y = slice_with_nans(ind, data, offset)
    ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, label=label)

    return ax


def plot_shade_mask(ax, ind, mask, facecolor="gray", alpha=0.5):
    """Shade across x values where boolean mask is `True`

    Args
    ----
    ax: pyplot.ax
        Axes object to plot with a shaded region
    ind: ndarray
        The indices to use for the x-axis values of the data
    mask: ndarray
        Boolean mask array to determine which regions should be shaded
    facecolor: matplotlib color
        Color of the shaded area

    Returns
    -------
    ax: pyplot.ax
        Axes object with the shaded region added
    """
    ymin, ymax = ax.get_ylim()
    ax.fill_between(ind, ymin, ymax, where=mask, facecolor=facecolor, alpha=alpha)
    return ax
