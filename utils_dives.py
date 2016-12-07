def calc_filter_dp(depths_m, cutoff, fs):
    '''Calculate the delta depth over time and filter to cuttoff frequency'''
    import numpy
    import scipy.signal

    from biotelem.acc import accfilter

    # Nyquist frequency
    nyq = fs/2.0
    # Calculate normalized cutoff freq with nyquist f
    dp_w = cutoff / nyq

    # TODO butter IIR filter, change to FIR?
    depth_fs = numpy.hstack(([0], numpy.diff(depths_m))) * fs
    #b, a     = scipy.signal.butter(4, dp_w, btype='low')
    #dp       = scipy.signal.filtfilt(b, a, depth_fs)
    b, a = accfilter.butter_lowpass(cutoff, fs)
    dp = accfilter.butter_apply(b, a, depth_fs)

    return dp


def finddives2(depths, min_dive_thresh=10):
    import numpy

    import utils

    condition = depths > min_dive_thresh
    ind_start, ind_end = utils.contiguous_regions(condition)

    dive_mask = numpy.zeros(len(depths), dtype=bool)
    depth_mean = numpy.zeros(len(ind_start))
    depth_comp = numpy.zeros(len(ind_start))

    for dive in range(len(ind_start)):
        dive_mask[ind_start[dive]:ind_end[dive]] = True
        dive_depths      = depths[ind_start[dive]:ind_end[dive]]
        depth_mean[dive] = numpy.mean(dive_depths)
        # Supoosedly time of deepest dive... doesn't appear to be that
        depth_comp[dive] = numpy.mean(1 + (1/(0.1*dive_depths)))

    dive_ind = numpy.vstack([ind_start, ind_end, depth_mean, depth_comp]).T

    # Filter any dives with an endpoint with an index beyond bounds of array
    dive_ind = dive_ind[dive_ind[:,1]<len(depths), :]

    return dive_ind, dive_mask


def get_des_asc2(depths, dive_mask, pitch, cutoff, fs, order=5):
    import numpy
    import utils_signal

    asc = numpy.zeros(len(depths), dtype=bool)
    des = numpy.zeros(len(depths), dtype=bool)

    # TODO and deg threshold, i.e. > 30
    #asc[dive_mask] = pitch[dive_mask] > 0
    #des[dive_mask] = pitch[dive_mask] < 0

    b, a = utils_signal.butter_filter(cutoff, fs, order, 'low')
    dfilt = utils_signal.butter_apply(b, a, depths)

    dp = numpy.hstack([numpy.diff(dfilt), 0])

    asc[dive_mask] = dp[dive_mask] < 0
    des[dive_mask] = dp[dive_mask] > 0

    # Remove descents/ascents withough a corresponding ascent/descent
    des, asc = rm_incomplete_des_asc(des, asc)

    return des, asc


def rm_incomplete_des_asc(des, asc):
    '''Remove descents-ascents that have no corresponding ascent-descent'''
    import utils

    # Get start/stop indices for descents and ascents
    des_start, des_stop = utils.contiguous_regions(des)
    asc_start, asc_stop = utils.contiguous_regions(asc)

    des = utils.rm_regions(des, asc, des_start, des_stop)
    asc = utils.rm_regions(asc, des, asc_start, asc_stop)

    return des, asc


def get_bottom(depths, des, asc):
    import numpy

    import utils

    # Get start/stop indices for descents and ascents
    des_start, des_stop = utils.contiguous_regions(des)
    asc_start, asc_stop = utils.contiguous_regions(asc)

    # Bottom time is at stop of descent until start of ascent
    bottom_len = min(len(des_stop), len(asc_start))
    bottom_start = des_stop[:bottom_len]
    bottom_stop  = asc_start[:bottom_len]

    bottom = numpy.zeros((len(bottom_start),4), dtype=float)

    # Time (seconds) at start of bottom phase/end of descent
    bottom[:,0] = bottom_start

    # Depth (m) at start of bottom phase/end of descent
    bottom[:,1] = depths[bottom_start]

    # Time (seconds) at end of bottom phase/start of asscent
    bottom[:,2] = bottom_stop

    # Depth (m) at end of bottom phase/start of descent
    bottom[:,3] = depths[bottom_stop]

    return bottom


def get_phase(depths, des, asc):
    '''get the directional phase sign for each sample in depths

    Args
    ----
    asc: numpy.ndarray, shape (n,)
        boolean mask of values where animal is ascending
    asc: numpy.ndarray, shape(n,)
        boolean mask of values where animal is descending

    Returns
    -------
    phase: numpy.ndarray, shape (n,)
        signed integer array with 0: neither ascending/descending, 1:
        ascending, -1: descending.
    '''
    import numpy

    phase = numpy.zeros(len(depths), dtype=int)

    phase[asc] =  1
    phase[des] = -1

    return phase


def create_dive_summary(dive_ind):
    '''Create a numpy array with summary values of dives

    Args
    ----
    dive_ind: numpy.ndarray
      Dive table

    Returns
    -------
    D: numpy.ndarray
      table of dive summary files

    Notes
    -----
    Implement as record array:
    D = numpy.zeros((n_dives)), dtype=dtypes)

    where:
    dtype = numpy.dtype([('start_time', int),  # start in sec since tag on time
                         ('end_time', int),    # end in sec since tag on time
                         ('duration', int),    # dive duration in sec
                         ('surface', int),     # post-dive surface duration in sec
                         ('max_time', int),    # time of deepest point
                         ('max_depth', float), # maximum dive depth of each dive
                         ('dive_id', int),     # dive ID number
                         ])
    '''
    import numpy

    n_dives = len(dive_ind[:, 0])

    D = numpy.zeros((n_dives, 7))

    # start in sec since tag on time
    D[:, 0] = dive_ind[:, 0]

    # end in sec since tag on time
    D[:, 1] = dive_ind[:, 1]

    # dive duration in sec
    D[:, 2] = dive_ind[:, 1] - dive_ind[:, 0]

    # post-dive surface duration in sec
    D[:, 3] = numpy.hstack((dive_ind[1:, 0] - dive_ind[0:-1, 1], [numpy.nan]))

    # time of deepest point
    D[:, 4] = dive_ind[:, 3]

    # maximum dive depth of each dive
    D[:, 5] = dive_ind[:, 2]

    # dive ID number
    D[:, 6] = numpy.arange(n_dives)

    return D


#def finddives(depths_m, fs, thresh=10, surface=1, findall=False):
#    '''Find time cues for the edges of dives.
#
#    Args
#    ----
#    depths_m: 1-D ndarray
#        is the depth time series in meters, sampled at `fs` Hz.
#
#    thresh: float
#        is the threshold in `m` at which to recognize a dive - dives more
#        shallow than `thresh` will be ignored. The default value for `thresh`
#        is `10m`.
#
#    surface: int, optional
#        is the depth in meters at which it is considered that the animal has
#        reached the surface. Default value is 1.
#
#    findall: bool, optional
#        force algorithm to include incomplete dives. `findall` = 1 forces the
#        algorithm to include incomplete dives at the start and end of the
#        record. Default is 0
#
#    Returns
#    -------
#    T: nx6 ndarray
#        is the matrix of cues with columns:
#
#        [start_idx, end_idx, max_depth, max_depth_idx, mean_depth, mean_compression]
#
#        If there are n dives deeper than thresh in `depths_m`, then T will be
#        an nx6 matrix. Partial dives at the beginning or end of the recording
#        will be ignored - only dives that start and end at the surface will
#        appear in T.
#    '''
#    import numpy
#
#    import utils
#
#    if fs > 1000:
#        raise SystemError('Suspicious fs of {} Hz - check '.format(round(fs)))
#
#    search_len = 20
#    dp_thresh  = 0.25
#    # Cutoff frequency
#    dp_lowpass = 0.5
#    # TODO remove or include somehow
#    cutoff = 0.15
#
#    # first remove any NaN at the start of depths_m
#    # (these are used to mask bad data points and only occur in a few data sets)
#    idx_good   = ~numpy.isnan(depths_m)
#    depths_m   = depths_m[idx_good]
#    t_good     = (min(idx_good) - 1) / fs
#
#    condition = depths_m > thresh
#    t_on, t_off = utils.contiguous_regions(condition)
#
#    # TODO needed?
#    ## truncate dive list to only dives with starts and stops in the record
#    #t_on  = t_on[:j]
#    #t_off = t_off[:j]
#
#    # filter vertical velocity to find actual surfacing moments
#    # TODO remove
#
#    dp = calc_filter_dp(depths_m, cutoff, fs)
#
#    # Search for surface events
#    dive_max = numpy.zeros((2, len(t_on)))
#    for i in range(len(t_on)):
#        # for each t_on, look back to find last time whale was at the surface
#        ind = t_on[i] - numpy.arange(round(search_len*fs), 0, -1)
#        ind = ind[ind >= 0]
#        try:
#            idx_i   = numpy.max(numpy.where(dp[ind] < dp_thresh)[0])
#            t_on[i] = ind[idx_i]
#        except ValueError:
#            t_on[i] = 0
#
#        # for each t_off, look forward to find next time whale is at the surface
#        ind = t_off[i] + numpy.arange(round(search_len*fs))
#        ind = ind[ind <= len(dp)-1]
#        try:
#            idx_i    = numpy.min(numpy.where(dp[ind] > -dp_thresh)[0])
#            t_off[i] = ind[idx_i]
#        except ValueError:
#            t_on[i] = len(dp)-1
#
#        # Get max dive depth, and index of max dive depth
#        dm   = numpy.max(depths_m[t_on[i]:t_off[i]])
#        km   = numpy.argmax(depths_m[t_on[i]:t_off[i]])
#
#        # Append `dm` to front of derived dive_max array
#        # TODO remove -1 from t_on + km -1 for python indexing?
#        dive_max[:, i] = numpy.hstack((dm, (t_on[i] + km)/fs + t_good))
#
#    # Raise error if dives not found in depth data
#    if len(t_on) < 1:
#        raise ValueError('No dives found in depth data. '
#                         'len(t_on)={}'.format(t_on))
#
#    # measure dive statistics
#    depth_mean = numpy.zeros(len(t_on))
#    depth_comp = numpy.zeros(len(t_on))
#
#    for i in range(len(t_on)):
#        depth_dive    = depths_m[t_on[i]:t_off[i]]
#        depth_mean[i] = numpy.mean(depth_dive)
#        depth_comp[i] = numpy.mean(1 + ((0.1*depth_dive) ** (- 1)))
#
#    # assemble output
#    t_mod = (numpy.vstack((t_on, t_off)) / fs) + t_good
#    T = numpy.vstack((t_mod.astype(int), dive_max, depth_mean, depth_comp)).T
#
#    return T


#def get_des_asc(depths, T, pitch, fs_a, min_dive_def=None, manual=False):
#    '''Return indices for descent and ascent periods of dives in T
#
#    3.1 quick separation of descent and ascent phases
#    '''
#    import numpy
#
#    import utils
#
#    # TODO check right version of fs used, fs_d?
#
#    print('\nGet descent and ascents from depths...')
#
#    # Init bottom, phase summary arrays
#    phase    = numpy.zeros(len(depths))
#    phase[:] = numpy.nan
#    bottom   = numpy.zeros((T.shape[0], 4))
#
#    # Init descent, ascent lists of indices
#    DES = list()
#    ASC = list()
#
#    # Index positions of bad dives to remove from T
#    bad_dives = list()
#
#    # If min_dive_def passed, get depths greater than `min_dive_def`
#    if min_dive_def:
#        # TODO 0.75 factor, pass as argument
#        depth_mask = depths < (min_dive_def * .75)
#    # Else include all depths in mask
#    else:
#        depth_mask = numpy.zeros(len(depths), dtype=bool)
#
#    # Get start, end indexs and dive stats for each dive
#    for dive in range(len(T)):
#        # get list of indices to select the whole dive
#        # multiply by acc sampling rate to scale indices
#        idx_start = (fs_a * T[dive, 0]).round()
#        idx_end = (fs_a * T[dive, 1]).round()
#        ind = numpy.arange(idx_start, idx_end, dtype=int)
#
#        # Convert kk indices to boolean mask
#        dive_mask = numpy.zeros(depths.size, dtype=bool)
#        dive_mask[ind] = True
#
#        # Omit depths less than `min_dive_def`
#        dive_mask[depth_mask] = False
#
#        # If passed add dive mask with dives where depth > min_dive_def
#        try:
#            # Find first index after diving below min_dive_def
#            # (pitch is positive)
#            end_pitch_mask = numpy.rad2deg(pitch[dive_mask]) > 0
#
#            end_pitch = numpy.where(end_pitch_mask)[0][0]
#            end_des   = int((end_pitch + (T[dive, 0] * fs_a)).round())
#
#            # Find last index before diving above min_dive_def
#            # (pitch is negative)
#            start_pitch_mask = numpy.rad2deg(pitch[dive_mask]) < 0
#
#            start_pitch = numpy.where(start_pitch_mask)[0][-1]
#            start_asc   = int((start_pitch + (T[dive, 0] * fs_a)).round())
#
#            if manual==False:
#                # selects the whole descent phase
#                des = numpy.arange((fs_a * T[dive, 0]).round(), end_des,
#                                   dtype=int)
#
#                # selects the whole ascent phase
#                asc = numpy.arange(start_asc, (fs_a * T[dive, 1]).round(),
#                                   dtype=int)
#            elif manual==True:
#                # TODO implement plotting
#                import warnings
#                warnings.warn('Manual dive descent/ascent selection has '
#                              'not been implemented. Proceeding with '
#                              'whole descent/ascent phase indicies')
#
#                # selects the whole descent phase
#                des = numpy.arange((fs_a * T[dive, 0]).round(), end_des,
#                                   dtype=int)
#
#                # selects the whole ascent phase
#                asc = numpy.arange(start_asc, (fs_a * T[dive, 1]).round(),
#                                   dtype=int)
#
#                # if you want to do it manually as some times there is a
#                # small ascent where pitch angle first goes to zero & last
#                # goes to zero in the ascent
#
#                # phase during the descent and a small descent phase during
#                # the ascent.
#                #     figure
#                #     # plott plots sensor data against a time axis
#                #     ax(1)=subplot(211) plott(depths[ind],fs_a)
#                #     ax(2)=subplot(212) plott(pitch[ind]*180/pi,fs_a,0)
#                #     # links x axes of the subplots for zoom/pan
#                #     linkaxes(ax, 'x')
#
#                #     # click on where the pitch angle first goes to zero
#                #     # in the descent and last goes to zero in the ascent
#                #     [x,y]=ginput(2)
#                #     des=round(x[1])/fs_a+T[dive,0]
#                #     asc=round(x[2])/fs_a+T[dive,0]
#
#            # Concatenate lists
#            if dive == 0:
#                DES = des
#                ASC = asc
#            else:
#                DES = numpy.hstack([DES, des])
#                ASC = numpy.hstack([ASC, asc])
#
#            phase[ind[ind < end_des]] = -1
#            phase[ind[(ind < start_asc) & (ind > end_des)]] = 0
#            phase[ind[ind > start_asc]] = 1
#
#            # Time in seconds at the start of bottom phase
#            # (end of descent)
#            bottom[dive, 0] = end_des / fs_a
#
#            # Depth in m at the start of the bottom phase
#            # (end of descent phase)
#            bottom[dive, 1] = depths[end_des]
#
#            # Time in seconds at the end of bottom phase
#            # (start of descent)
#            bottom[dive, 2] = start_asc / fs_a
#
#            # Depth in m at the end of the bottom phase
#            # (start of descent phase)
#            bottom[dive, 3] = depths[start_asc]
#
#        # If acc signal does not match depth movement, remove dive
#        except IndexError:
#            print('Empty pitch array, likely all positive/negative.')
#            # remove invalid dive from summary table
#            bad_dives.append(dive)
#            continue
#
#    print('')
#
#    T = numpy.delete(T, bad_dives, 0)
#
#    return T, DES, ASC, phase, bottom


#def get_dive_mask(depths, T, fs):
#    '''Get boolean mask of values in depths that are dives'''
#    import numpy
#
#    dive_mask = numpy.zeros(depths.size, dtype=bool)
#    for i in range(T.shape[0]):
#        idx_start = int(round(T[i, 0] * fs))
#        idx_end  = int(round(T[i, 1] * fs))
#        dive_mask[idx_start:idx_end] = True
#
#    return dive_mask


#def select_last_dive(depths, pitch, pitch_lf, T, fs):
#    '''Get user selection of last dive to start index from
#
#    `isdive` renamed `dive_mask`
#    '''
#    import copy
#    import numpy
#
#    import utils
#    import utils_dives
#    import utils_plot
#
#    # Setup subplots
#    utils_plot.plot_dives_pitch(depths, dive_mask, des, asc, pitch, pitch_lf)
#
#    # Get user input for first index position of last dive to determine
#    x = utils.recursive_input('first index of last dive to use', int)
#
#    # TODO .m code took indices of whole dive, then first of those used
#    #nn = numpy.where(T[:, 0] < x[0]/fs)[0][-1]
#    nn = int(numpy.where(T[:, 0] < x/fs)[0][-1])
#
#    return nn
