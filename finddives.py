
def finddives(depths_m, fs, thresh=10, surface=1, findall=False):
    '''Find time cues for the edges of dives.

    depths_m: is the depth time series in meters, sampled at `fs` Hz.

    thresh:  is the threshold in `m` at which to recognize a dive - dives more
             shallow than `thresh` will be ignored. The default value for
             `thresh` is `10m`.

    surface: is the depth in meters at which it is considered that the
             animal has reached the surface. Default value is 1.

    findall: force algorithm to include incomplete dives. `findall` = 1 forces
             the algorithm to include incomplete dives at the start and end of
             the record. Default is 0

    T:       is the matrix of cues with columns:
             [start_cue, end_cue, max_depth, cue_at_max_depth, mean_depth,
             mean_compression]

             If there are n dives deeper than thresh in `depths_m`, then T will
             be an nx6 matrix. Partial dives at the beginning or end of the
             recording will be ignored - only dives that start and end at the
             surface will appear in T.
    '''
    import numpy
    import scipy.signal

    if fs > 1000:
        raise SystemError('Suspicious fs of {} Hz - check '.format(round(fs)))

    search_len = 20
    dp_thresh  = 0.25
    dp_lowpass = 0.5

    # first remove any NaN at the start of depths_m
    # (these are used to mask bad data points and only occur in a few data sets)
    idx_good   = ~numpy.isnan(depths_m)
    depths_m   = depths_m[idx_good]
    t_good     = (min(numpy.where(idx_good)) - 1) / fs

    # find threshold crossings and surface times
    t_thresh = numpy.where(numpy.diff(depths_m > thresh) > 0)[0]
    t_surf   = numpy.where(depths_m < surface)
    t_on     = numpy.zeros(len(t_thresh))
    t_off    = numpy.zeros(len(t_thresh))

    # Test if numpy array is empty
    notempty = lambda x:~x.size != 0

    # sort through threshold crossings to find valid dive start and end points
    j = 0
    for i in range(len(t_thresh)):
        if all(t_thresh[i] > t_off):
            idx_s0 = numpy.where(t_surf < t_thresh[i])
            idx_s1 = numpy.where(t_surf > t_thresh[i])
            # Findall forces incomplete dives to included, see docstring
            if findall or (notempty(idx_s0) and notempty(idx_s1)):
                j += 1

                if idx_s0 == 0:
                    t_on[j] = 1
                else:
                    t_on[j] = max(t_surf[idx_s0])

                if idx_s1 == 0:
                    t_off[j] = len(depths_m)
                else:
                    t_off[j] = min(t_surf[idx_s1])

    # truncate dive list to only dives with starts and stops in the record
    t_on  = t_on[:j]
    t_off = t_off[:j]

    # filter vertical velocity to find actual surfacing moments
    depth_fs = numpy.hstack(([0], numpy.diff(depths_m)]))*fs
    b, a     = scipy.signal.butter(4, dp_lowpass / (fs / 2), btype='low')
    dp       = scipy.signal.filtfilt(b, a, depth_fs)

    # Search for surface events
    dive_max = numpy.zeros((len(t_on), 2))
    for i in range(len(t_on)):
        # for each t_on, look back to find last time whale was at the surface
        ind    = t_on[i] + numpy.asarray(range(round(search_len * fs)), 0, -1)
        ind    = ind[numpy.where(ind > 0)]
        idx_i  = max(numpy.where(dp[ind] < dp_thresh))

        # for each t_off, look forward to find next time whale is at the surface
        t_on[i] = ind[idx_i]
        ind     = t_off[i] + numpy.asarray(range(round(search_len*fs)))
        ind     = ind[numpy.where(ind <= len(depths_m))]
        idx_i   = min(numpy.where(dp[ind] > -dp_thresh))

        t_off[i] = ind[idx_i]
        dm, km   = max(depths_m[t_on[i]:t_off[i]])

        # Append `dm` to front of derived dive_max array
        dive_max[k, :] = numpy.hstack((dm, (t_on[i] + km - 1)/fs + t_good))

    # measure dive statistics
    depth_mean = numpy.zeros(len(t_on))
    depth_comp = numpy.zeros(len(t_on))

    for i in range(len(t_on)):
        depth_dive    = depths_m[t_on[i]:t_off[i]]
        depth_mean[i] = numpy.mean(depth_dive)
        depth_comp[i] = numpy.mean(1 + ((0.1*depth_dive) ** (- 1)))

    # assemble output
    t_mod = (numpy.vstack((t_on, t_off)) / fs) + t_good
    T = numpy.vstack((t_mod, dive_max, depth_mean, depth_comp)).T

    return T
