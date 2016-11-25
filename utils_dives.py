def calc_filter_dp(depths_m, cutoff, fs):
    '''Calculate the delta depth over time and filter to cuttoff frequency'''
    import numpy
    import scipy.signal

    from biotelem.acc import accfilter

    # Nyquist frequency
    nyq = fs/2.0
    # Calculate normalized cutoff freq with nyquist f
    dp_w = cutoff / nyq

    depth_fs = numpy.hstack(([0], numpy.diff(depths_m))) * fs
    #b, a     = scipy.signal.butter(4, dp_w, btype='low')
    #dp       = scipy.signal.filtfilt(b, a, depth_fs)
    b, a = accfilter.butter_lowpass(cutoff, fs)
    dp = accfilter.butter_apply(b, a, depth_fs)

    return dp


def finddives(depths_m, fs, thresh=10, surface=1, findall=False):
    '''Find time cues for the edges of dives.

    Args
    ----
    depths_m: 1-D ndarray
        is the depth time series in meters, sampled at `fs` Hz.

    thresh: float
        is the threshold in `m` at which to recognize a dive - dives more
        shallow than `thresh` will be ignored. The default value for `thresh`
        is `10m`.

    surface: int, optional
        is the depth in meters at which it is considered that the animal has
        reached the surface. Default value is 1.

    findall: bool, optional
        force algorithm to include incomplete dives. `findall` = 1 forces the
        algorithm to include incomplete dives at the start and end of the
        record. Default is 0

    Returns
    -------
    T: nx6 ndarray
        is the matrix of cues with columns:

        [start_idx, end_idx, max_depth, max_depth_idx, mean_depth, mean_compression]

        If there are n dives deeper than thresh in `depths_m`, then T will be
        an nx6 matrix. Partial dives at the beginning or end of the recording
        will be ignored - only dives that start and end at the surface will
        appear in T.
    '''
    import numpy

    import utils

    if fs > 1000:
        raise SystemError('Suspicious fs of {} Hz - check '.format(round(fs)))

    search_len = 20
    dp_thresh  = 0.25
    # Cutoff frequency
    dp_lowpass = 0.5
    # TODO remove or include somehow
    cutoff = 0.15

    # first remove any NaN at the start of depths_m
    # (these are used to mask bad data points and only occur in a few data sets)
    idx_good   = ~numpy.isnan(depths_m)
    depths_m   = depths_m[idx_good]
    t_good     = (min(idx_good) - 1) / fs

    condition = depths_m > thresh
    t_on, t_off = utils.contiguous_regions(condition)

    # TODO needed?
    ## truncate dive list to only dives with starts and stops in the record
    #t_on  = t_on[:j]
    #t_off = t_off[:j]

    # filter vertical velocity to find actual surfacing moments
    # TODO remove

    dp = calc_filter_dp(depths_m, cutoff, fs)

    # Search for surface events
    dive_max = numpy.zeros((2, len(t_on)))
    for i in range(len(t_on)):
        # for each t_on, look back to find last time whale was at the surface
        ind = t_on[i] - numpy.arange(round(search_len*fs), 0, -1)
        ind = ind[ind >= 0]
        try:
            idx_i   = numpy.max(numpy.where(dp[ind] < dp_thresh)[0])
            t_on[i] = ind[idx_i]
        except ValueError:
            t_on[i] = 0

        # for each t_off, look forward to find next time whale is at the surface
        ind = t_off[i] + numpy.arange(round(search_len*fs))
        ind = ind[ind <= len(dp)-1]
        try:
            idx_i    = numpy.min(numpy.where(dp[ind] > -dp_thresh)[0])
            t_off[i] = ind[idx_i]
        except ValueError:
            t_on[i] = len(dp)-1

        # Get max dive depth, and index of max dive depth
        dm   = numpy.max(depths_m[t_on[i]:t_off[i]])
        km   = numpy.argmax(depths_m[t_on[i]:t_off[i]])

        # Append `dm` to front of derived dive_max array
        # TODO remove -1 from t_on + km -1 for python indexing?
        dive_max[:, i] = numpy.hstack((dm, (t_on[i] + km)/fs + t_good))

    # Raise error if dives not found in depth data
    if len(t_on) < 1:
        raise ValueError('No dives found in depth data. '
                         'len(t_on)={}'.format(t_on))

    # measure dive statistics
    depth_mean = numpy.zeros(len(t_on))
    depth_comp = numpy.zeros(len(t_on))

    for i in range(len(t_on)):
        depth_dive    = depths_m[t_on[i]:t_off[i]]
        depth_mean[i] = numpy.mean(depth_dive)
        depth_comp[i] = numpy.mean(1 + ((0.1*depth_dive) ** (- 1)))

    # assemble output
    t_mod = (numpy.vstack((t_on, t_off)) / fs) + t_good
    T = numpy.vstack((t_mod.astype(int), dive_max, depth_mean, depth_comp)).T

    return T


def get_des_asc(depths, T, pitch, fs_a, min_dive_def=None, manual=False):
    '''Return indices for descent and ascent periods of dives in T

    3.1 quick separation of descent and ascent phases
    '''
    import numpy

    import utils

    # TODO check right version of fs used, fs_d?

    print('\nGet descent and ascents from depths...')

    # Init bottom, phase summary arrays
    phase    = numpy.zeros(len(depths))
    phase[:] = numpy.nan
    bottom   = numpy.zeros((T.shape[0], 4))

    # Init descent, ascent lists of indices
    DES = list()
    ASC = list()

    # Index positions of bad dives to remove from T
    bad_dives = list()

    # Get depths greater than `min_dive_def`
    if min_dive_def:
        depth_mask = depths > (min_dive_def * .75)

    # Get start, end indexs and dive stats for each dive
    for dive in range(len(T)):
        # get list of indices to select the whole dive
        # multiply by acc sampling rate to scale indices
        idx0 = numpy.floor(fs_a * T[dive, 0])
        idx1 = numpy.floor(fs_a * T[dive, 1])
        ind = numpy.arange(idx0, idx1, dtype=int)

        # Convert kk indices to boolean mask
        dive_mask = numpy.zeros(depths.size, dtype=bool)
        dive_mask[ind] = True

        # If passed add dive mask with dives where depth > min_dive_def
        if min_dive_def:
            dive_mask = dive_mask & depth_mask

        try:
            # Find first index after diving below min_dive_def
            # (pitch is positive)
            end_pitch_mask = numpy.rad2deg(pitch[dive_mask]) > 0
            end_pitch      = numpy.where(end_pitch_mask)[0][0]
            end_des        = round(end_pitch + (T[dive, 0] * fs_a))

            # Find last index before diving above min_dive_def
            # (pitch is negative)
            start_pitch_mask = numpy.rad2deg(pitch[dive_mask]) < 0
            start_pitch      = numpy.where(start_pitch_mask)[0][-1]
            start_asc        = round(start_pitch + (T[dive, 0] * fs_a))

            if manual==False:
                # selects the whole descent phase
                des = list(range(round(fs_a * T[dive, 0]), end_des))

                # selects the whole ascent phase
                asc = list(range(start_asc, round(fs_a * T[dive, 1])))
            elif manual==True:
                # TODO implement plotting
                import warnings
                warnings.warn('Manual dive descent/ascent selection has '
                              'not been implemented. Proceeding with '
                              'whole descent/ascent phase indicies')

                # selects the whole descent phase
                des = list(range(round(fs_a * T[dive, 0]), end_des))

                # selects the whole ascent phase
                asc = list(range(start_asc, round(fs_a * T[dive, 1])))

                # if you want to do it manually as some times there is a
                # small ascent where pitch angle first goes to zero & last
                # goes to zero in the ascent

                # phase during the descent and a small descent phase during
                # the ascent.
                #     figure
                #     # plott plots sensor data against a time axis
                #     ax(1)=subplot(211) plott(depths[ind],fs_a)
                #     ax(2)=subplot(212) plott(pitch[ind]*180/pi,fs_a,0)
                #     # links x axes of the subplots for zoom/pan
                #     linkaxes(ax, 'x')

                #     # click on where the pitch angle first goes to zero
                #     # in the descent and last goes to zero in the ascent
                #     [x,y]=ginput(2)
                #     des=round(x[1])/fs_a+T[dive,0]
                #     asc=round(x[2])/fs_a+T[dive,0]

            # Concatenate lists
            DES += des
            ASC += asc

            phase[ind[ind < end_des]] = -1
            phase[ind[(ind < start_asc) & (ind > end_des)]] = 0
            phase[ind[ind > start_asc]] = 1

            # Time in seconds at the start of bottom phase
            # (end of descent)
            bottom[dive, 0] = end_des / fs_a

            # Depth in m at the start of the bottom phase
            # (end of descent phase)
            bottom[dive, 1] = depths[end_des]

            # Time in seconds at the end of bottom phase
            # (start of descent)
            bottom[dive, 2] = start_asc / fs_a

            # Depth in m at the end of the bottom phase
            # (start of descent phase)
            bottom[dive, 3] = depths[start_asc]

        # If acc signal does not match depth movement, remove dive
        except IndexError:
            print('Empty pitch array, likely all positive/negative.')
            # remove invalid dive from summary table
            bad_dives.append(dive)
            continue

    print('')

    T = numpy.delete(T, bad_dives, 0)

    return T, DES, ASC, phase, bottom


def get_dive_mask(depths, T, fs):
    '''Get boolean mask of values in depths that are dives'''
    import numpy

    isdive = numpy.zeros(depths.size, dtype=bool)
    for i in range(T.shape[0]):
        isdive[round(T[i, 0] * fs) : round(T[i, 1] * fs)] = True

    return isdive
