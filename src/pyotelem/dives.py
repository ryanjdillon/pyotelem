
def finddives2(depths, min_dive_thresh=10):
    '''Find dives in depth data below a minimum dive threshold

    Args
    ----
    depths: ndarray
        Datalogger depth measurements
    min_dive_thresh: float
        Minimum depth threshold for which to classify a dive

    Returns
    -------
    dives: ndarray
        Dive summary information in a numpy record array

        *Columns*:

        * dive_id
        * start_idx
        * stop_idx
        * dive_dur
        * depth_max
        * depth_max_i
        * depth_min
        * depth_min_i
        * depth_mean
        * comp_mean

    dive_mask: ndarray
        Boolean mask array over depth data. Cells with `True` are dives and
        cells with `False` are not.
    '''
    import numpy
    import pandas

    from . import utils

    # Get start and stop indices for each dive above `min_dive_thresh`
    condition = depths > min_dive_thresh
    ind_start, ind_end = utils.contiguous_regions(condition)

    n_dives = len(ind_start)

    dive_mask = numpy.zeros(len(depths), dtype=bool)

    dtypes = numpy.dtype([('dive_id', int),
                          ('start_idx', int),
                          ('stop_idx', int),
                          ('dive_dur', int),
                          ('depth_max', float),
                          ('depth_max_idx', float),
                          ('depth_min', float),
                          ('depth_min_idx', float),
                          ('depth_mean', float),
                          ('comp_mean', float),])

    dive_data = numpy.zeros(n_dives, dtype=dtypes)

    for i in range(n_dives):
        dive_mask[ind_start[i]:ind_end[i]] = True
        dive_depths                   = depths[ind_start[i]:ind_end[i]]
        dive_data['dive_id'][i]       = i
        dive_data['start_idx'][i]     = ind_start[i]
        dive_data['stop_idx'][i]      = ind_end[i]
        dive_data['dive_dur'][i]      = ind_end[i] - ind_start[i]
        dive_data['depth_max'][i]     = dive_depths.max()
        dive_data['depth_max_idx'][i] = numpy.argmax(dive_depths)
        dive_data['depth_min'][i]     = dive_depths.min()
        dive_data['depth_min_idx'][i] = numpy.argmin(dive_depths)
        dive_data['depth_mean'][i]    = numpy.mean(dive_depths)
        # TODO Supposedly time of deepest dive... doesn't appear to be that
        dive_data['comp_mean'][i]     = numpy.mean(1 + (1/(0.1*dive_depths)))

    # Filter any dives with an endpoint with an index beyond bounds of array
    dive_data = dive_data[dive_data['stop_idx'] < len(depths)]

    # Create pandas data frame with following columns, init'd with nans
    dives = pandas.DataFrame(dive_data)

    return dives, dive_mask


def get_des_asc2(depths, dive_mask, pitch, cutoff, fs, order=5):
    '''Get boolean masks of descents and ascents in the depth data

    Args
    ----
    dive_mask: ndarray
        Boolean mask array over depth data. Cells with `True` are dives and
        cells with `False` are not.
    pitch: ndarray
        Pitch angle in radians
    cutoff: float
        Cutoff frequency at which signal will be filtered
    fs: float
        Sampling frequency
    order: int
        Order of butter filter to apply

    Returns
    -------
    des_mask: ndarray
        Boolean mask of descents in the depth data
    asc_mask: ndarray
        Boolean mask of ascents in the depth data
    '''
    import numpy

    from . import dsp

    asc_mask = numpy.zeros(len(depths), dtype=bool)
    des_mask = numpy.zeros(len(depths), dtype=bool)

    b, a = dsp.butter_filter(cutoff, fs, order, 'low')
    dfilt = dsp.butter_apply(b, a, depths)

    dp = numpy.hstack([numpy.diff(dfilt), 0])

    asc_mask[dive_mask] = dp[dive_mask] < 0
    des_mask[dive_mask] = dp[dive_mask] > 0

    # Remove descents/ascents withough a corresponding ascent/descent
    des_mask, asc_mask = rm_incomplete_des_asc(des_mask, asc_mask)

    return des_mask, asc_mask


def rm_incomplete_des_asc(des_mask, asc_mask):
    '''Remove descents-ascents that have no corresponding ascent-descent

    Args
    ----
    des_mask: ndarray
        Boolean mask of descents in the depth data
    asc_mask: ndarray
        Boolean mask of ascents in the depth data

    Returns
    -------
    des_mask: ndarray
        Boolean mask of descents with erroneous regions removed
    asc_mask: ndarray
        Boolean mask of ascents with erroneous regions removed
    '''
    from . import utils

    # Get start/stop indices for descents and ascents
    des_start, des_stop = utils.contiguous_regions(des_mask)
    asc_start, asc_stop = utils.contiguous_regions(asc_mask)

    des_mask = utils.rm_regions(des_mask, asc_mask, des_start, des_stop)
    asc_mask = utils.rm_regions(asc_mask, des_mask, asc_start, asc_stop)

    return des_mask, asc_mask


def get_bottom(depths, des_mask, asc_mask):
    '''Get boolean mask of regions in depths the animal is at the bottom

    Args
    ----
    des_mask: ndarray
        Boolean mask of descents in the depth data
    asc_mask: ndarray
        Boolean mask of ascents in the depth data

    Returns
    -------
    BOTTOM: ndarray (n,4)
        Indices and depths for when the animal is at the bottom

        *Index positions*:

        0. start ind
        1. depth at start
        2. stop ind
        3. depth at stop
    '''
    import numpy

    from . import utils

    # Get start/stop indices for descents and ascents
    des_start, des_stop = utils.contiguous_regions(des_mask)
    asc_start, asc_stop = utils.contiguous_regions(asc_mask)

    # Bottom time is at stop of descent until start of ascent
    bottom_len = min(len(des_stop), len(asc_start))
    bottom_start = des_stop[:bottom_len]
    bottom_stop  = asc_start[:bottom_len]

    BOTTOM = numpy.zeros((len(bottom_start),4), dtype=float)

    # Time (seconds) at start of bottom phase/end of descent
    BOTTOM[:,0] = bottom_start

    # Depth (m) at start of bottom phase/end of descent
    BOTTOM[:,1] = depths[bottom_start]

    # Time (seconds) at end of bottom phase/start of asscent
    BOTTOM[:,2] = bottom_stop

    # Depth (m) at end of bottom phase/start of descent
    BOTTOM[:,3] = depths[bottom_stop]

    return BOTTOM


def get_phase(n_samples, des_mask, asc_mask):
    '''Get the directional phase sign for each sample in depths

    Args
    ----
    n_samples: int
        Length of output phase array
    des_mask: numpy.ndarray, shape (n,)
        Boolean mask of values where animal is descending
    asc_mask: numpy.ndarray, shape(n,)
        Boolean mask of values where animal is ascending

    Returns
    -------
    phase: numpy.ndarray, shape (n,)
        Signed integer array values representing animal's dive phase

        *Phases*:

        *  0: neither ascending/descending
        *  1: ascending
        * -1: descending.
    '''
    import numpy

    phase = numpy.zeros(n_samples, dtype=int)
    phase[asc_mask] =  1
    phase[des_mask] = -1

    return phase
