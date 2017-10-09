'''
Glides and subglide plotting functions
'''
import matplotlib.pyplot as plt

from . import plotconfig as _plotconfig
from .plotconfig import _colors

def plot_glide_depths(depths, mask_tag_filt):
    '''Plot depth at glides

    Args
    ----
    depths: ndarray
        Depth values at each sensor sampling
    mask_tag_filt: ndarray
        Boolean mask to slice filtered sub-glides from tag data
    '''
    import numpy

    from . import plotutils

    fig, ax = plt.subplots()

    ax = plotutils.plot_noncontiguous(ax, depths, numpy.where(mask_tag_filt)[0])
    ax.invert_yaxis()

    plt.show()

    return None


def plot_sgls(mask_exp, depths, mask_tag_filt, sgls, mask_sgls_filt, Az_g_hf,
        idx_start=None, idx_end=None, path_plot=None, linewidth=0.5,
        leg_bbox=(1.23,1), clip_x=False):
    '''Plot sub-glides over depth and high-pass filtered accelerometer signal

    Args
    ----
    mask_exp: ndarray
        Boolean mask array to slice tag data to experimtal period
    depths: ndarray
        Depth values at each sensor sampling
    mask_tag_filt: ndarray
        Boolean mask to slice filtered sub-glides from tag data
    sgls: pandas.DataFrame
        Sub-glide summary information defined by `SGL` start/stop indices
    mask_sgls_filt: ndarray
        Boolean mask to slice filtered sub-glides from sgls data
    Az_g_hf: ndarray
        High-pass filtered, calibrated z-axis accelerometer data
    idx_start: int
        Sample index position where plot should begin
    idx_stop: int
        Sample index position where plot should stop
    path_plot: str
        Path and filename for figure to be saved
    linewidth: float
        Width of plot lines (Default: 0.5)
    clip_x: bool
        Swith to clip x-axis to the experimental period
    '''
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    import numpy

    from . import plotutils
    from .. import utils

    # Create experiment mask from specified start/end indices if passed
    if idx_start or idx_end:
        mask_exp = numpy.zeros(len(depths), dtype=bool)
        if idx_start and idx_end:
            mask_exp[idx_start:idx_end] = True
        elif idx_start:
            mask_exp[idx_start:ind_exp[-1]] = True
        elif idx_end:
            mask_exp[ind_exp[0]:idx_end] = True

    # Filter passed data to experimental period
    depths  = depths[mask_exp]
    Az_g_hf = Az_g_hf[mask_exp]

    # Create subglide indice groups for plotting
    sgl_ind    = numpy.where(mask_tag_filt & mask_exp)[0]
    notsgl_ind = numpy.where((~mask_tag_filt) & mask_exp)[0]
    # Create experiment indices from `mask_exp`
    ind_exp = numpy.where(mask_exp)[0]
    offset = 0
    plt_offset = ind_exp[0]

    # Clip values to within experimental period
    if clip_x:
        offset = ind_exp[0]
        ind_exp = ind_exp - offset
        sgl_ind    = sgl_ind - offset
        notsgl_ind = notsgl_ind - offset
        plt_offset = 0

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot glides
    c0, c1 = _colors[0:2]
    ax1 = plotutils.plot_noncontiguous(ax1, depths, sgl_ind, c0, 'Glides',
                                       offset=plt_offset,
                                       linewidth=linewidth*2)
    ax1 = plotutils.plot_noncontiguous(ax1, depths, notsgl_ind, c1, 'Stroking',
                                       offset=plt_offset, linewidth=linewidth,
                                       linestyle='--')

    # Plot HF Z-axis
    c0 = _colors[2]
    ax2.plot(ind_exp, Az_g_hf, color=c0, label='Z-axis HF Acc.',
            linewidth=linewidth)

    # Get dives within mask
    gg = sgls[mask_sgls_filt]

    # Get midpoint of dive occurance
    x = (gg['start_idx'] + (gg['stop_idx'] - gg['start_idx'])/2)
    x = x.values.astype(float)
    x_mask = (x-offset > ind_exp[0]) & (x-offset< ind_exp[-1])
    x = x[x_mask]

    # Get depth at midpoint
    if clip_x:
        x = x - offset
        ind_x = numpy.round(x).astype(int)
    else:
        ind_x = numpy.round(x - plt_offset).astype(int)
    y = depths[ind_x]

    # For each dive_id, sgl_id pair, create annotation string, apply
    dids = gg['dive_id'].values.astype(int)
    sids = numpy.array(gg.index)
    dids = dids[x_mask]
    sids = sids[x_mask]
    n = ['Dive:{}, SGL:{}'.format(did, sid) for did, sid in zip(dids, sids)]

    diff = ind_exp[1] - ind_exp[0]
    for i, txt in enumerate(n):
        # TODO semi-hardcoded dist for annotation
        ax1.annotate(txt, (x[i]+int(diff*16),y[i]))

    # Plot shaded areas where not sub-glides
    ax1 = plotutils.plot_shade_mask(ax1, ind_exp, ~mask_tag_filt[mask_exp],
            facecolor='#d9d9d9')
    ax2 = plotutils.plot_shade_mask(ax2, ind_exp, ~mask_tag_filt[mask_exp],
            facecolor='#d9d9d9')

    # Set x-axes limits
    for ax in [ax1, ax2]:
        ticks = ax.get_yticks()
        ax.set_ylim((ticks[0], ticks[-1]))
        if idx_start:
            xmin = idx_start
        else:
            xmin = ax.get_xlim()[0]
        if idx_end:
            xmax = idx_end
        else:
            xmax = ax.get_xlim()[1]
        if clip_x:
            xmin, xmax = xmin-offset, xmax-offset
        ax.set_xlim(xmin, xmax)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')

    # Update Depth subplot y-axis labels, limits, invert depth
    ax1.set_ylabel('Depth ($m$)')
    ymin = depths.min() - (depths.max()*0.01)
    ymax = depths.max() + (depths.max()*0.01)
    ax1.set_ylim((ymin, ymax))
    ax1.invert_yaxis()
    ax1.get_yaxis().set_label_coords(-0.09,0.5)

    # Update PRH subplot y labels, limits
    ax2.set_ylabel('Z-axis acceleration ($g$)')
    ax2.set_ylim((Az_g_hf.min(), Az_g_hf.max()))
    ax2.get_yaxis().set_label_coords(-0.09,0.5)

    # Scientific notation for ax1 `n_samples`
    ax1.set_xlabel('No. sensor samples')
    mf1 = ScalarFormatter(useMathText=True)
    mf1.set_powerlimits((-2,2))
    ax1.xaxis.set_major_formatter(mf1)

    # Convert n_samples to hourmin labels
    ax2.set_xlabel('Experiment duration ($min \, sec$)')
    mf2 = FuncFormatter(plotutils.nsamples_to_minsec)
    ax2.xaxis.set_major_formatter(mf2)

    # Create legends outside plot area
    leg1 = ax1.legend(bbox_to_anchor=leg_bbox)
    plt.tight_layout(rect=[0,0,0.8,1])

    # Save plot if `path_plot` passed
    if path_plot:
        import os
        fname = 'subglide_highlight'
        if idx_start:
            fname += '_start{}'.format(idx_start)
        if idx_end:
            fname+= '_stop{}'.format(idx_end)
        ext = '.eps'
        file_fig = os.path.join(path_plot, fname+ext)
        plt.savefig(file_fig, box='tight')

    plt.show()

    return None
