import matplotlib.pyplot as plt

from . import plotconfig as _plotconfig
from .plotconfig import _colors, _linewidth

def plot_glide_depths(depths, mask_tag_filt):
    '''Plot depth at glides'''
    import numpy

    from . import plotutils

    fig, ax = plt.subplots()

    ax = plotutils.plot_noncontiguous(ax, depths, numpy.where(mask_tag_filt)[0])
    ax.invert_yaxis()

    plt.show()

    return None


def plot_sgls(mask_exp, depths, mask_tag_filt, sgls, mask_sgls_filt, pitch_lf,
        roll_lf, heading_lf, idx_start=None, idx_end=None, path_plot=None):

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
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

    # Create experiment indices from `mask_exp`
    ind_exp = numpy.where(mask_exp)[0]

    # Filter passed data to experimental period
    depths      = depths[mask_exp]
    pitch_deg   = numpy.rad2deg(pitch_lf[mask_exp])
    roll_deg    = numpy.rad2deg(roll_lf[mask_exp])
    heading_deg = numpy.rad2deg(heading_lf[mask_exp])

    # Create subglide indice groups for plotting
    sgl_ind    = numpy.where(mask_tag_filt & mask_exp)[0]
    notsgl_ind = numpy.where((~mask_tag_filt) & mask_exp)[0]
    sgl_ind    = sgl_ind - ind_exp[0]
    notsgl_ind = notsgl_ind - ind_exp[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # Plot glides
    c0, c1 = _colors[0:2]
    ax1 = plotutils.plot_noncontiguous(ax1, depths, sgl_ind, c0, 'Glides')
    ax1 = plotutils.plot_noncontiguous(ax1, depths, notsgl_ind, c1, 'Stroking')

    # Plot PRH
    c0, c1, c2 = _colors[2:5]
    x = ind_exp - ind_exp[0]
    ax2.plot(x, pitch_deg, color=c0, label='Pitch', linewidth=_linewidth)
    ax2.plot(x, roll_deg, color=c1, label='Roll', linewidth=_linewidth)
    ax2.plot(x, heading_deg, color=c2, label='Heading', linewidth=_linewidth)

    # Get dives within mask
    gg = sgls[mask_sgls_filt]

    # Get midpoint of dive occurance
    x = (gg['start_idx'] + (gg['stop_idx'] - gg['start_idx'])/2)
    x = x.values.astype(float)
    x_mask = (x > ind_exp[0]) & (x < ind_exp[-1])
    x = x[x_mask]
    x = x - ind_exp[0]

    # Get depth at midpoint
    y = depths[numpy.round(x).astype(int)]

    # For each dive_id, sgl_id pair, create annotation string, apply
    dids = gg['dive_id'].values.astype(int)
    sids = numpy.array(gg.index)
    dids = dids[x_mask]
    sids = sids[x_mask]
    n = ['Dive:{}, SGL:{}'.format(did, sid) for did, sid in zip(dids, sids)]

    for i, txt in enumerate(n):
        ax1.annotate(txt, (x[i],y[i]))

    # Plot shaded areas where not sub-glides
    ax1 = plotutils.plot_shade_mask(ax1, ~mask_tag_filt[mask_exp])
    ax2 = plotutils.plot_shade_mask(ax2, ~mask_tag_filt[mask_exp])

    # Set x-axes limits
    for ax in [ax1, ax2]:
        ticks = ax.get_yticks()
        ax.set_ylim((ticks[0], ticks[-1]))
        if idx_start:
            xmin = idx_start - ind_exp[0]
        if idx_end:
            xmax = idx_end - ind_exp[0]
        ax.set_xlim(xmin, xmax)

    # Update Depth subplot y-axis labels, limits, invert depth
    ax1.yaxis.label.set_text('Depth ($m$)')
    ymin = depths.min() - (depths.max()*0.01)
    ymax = depths.max() + (depths.max()*0.01)
    print('depths', depths.min(), depths.max())
    print('ylim', ymin, ymax)
    ax1.set_ylim((ymin, ymax))
    ax1.invert_yaxis()
    ax1.get_yaxis().set_label_coords(-0.06,0.5)

    # Update PRH subplot y labels, limits
    ax2.yaxis.label.set_text('Angle ($\degree$)')
    ax2.xaxis.label.set_text('Experiment duration')
    deg_min = min([pitch_deg.min(), roll_deg.min(), heading_deg.min()])
    deg_max = max([pitch_deg.max(), roll_deg.max(), heading_deg.max()])
    ymin = -185
    ymax = 185
    ax2.set_ylim((ymin, ymax))
    ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.get_yaxis().set_label_coords(-0.06,0.5)

    # Convert n_samples to hourmin labels
    formatter = FuncFormatter(plotutils.nsamples_to_hourmin)
    ax2.xaxis.set_major_formatter(formatter)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)

    # Create legends outside plot area
    leg1 = ax1.legend(loc='upper right', bbox_to_anchor=(1.28,1))
    leg2 = ax2.legend(loc='upper right', bbox_to_anchor=(1.28,1))
    plt.tight_layout(rect=[0,0,0.8,1])

    # Save plot if `path_plot` passed
    if path_plot:
        import os
        fname = 'subglide_highlight'
        ext = '.png'
        file_fig = os.path.join(path_plot, fname+ext)
        plt.savefig(file_fig, box='tight')

    plt.show()

    return None
