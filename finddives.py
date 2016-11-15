def plot_two(dv0, dv1, p, dp, t_on, t_off):
    '''Plots depths and delta depths with dive start stop markers'''
    import matplotlib.pyplot as plt

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.title.set_text('Dives')

    x0   = t_on[dv0:dv1] - t_on[dv0]
    x1   = t_off[dv0:dv1] - t_on[dv0]

    # Extract start end depths
    y0_p = p[t_on[dv0:dv1]]
    y1_p = p[t_off[dv0:dv1]]

    # Extract start end delta depths
    y0_dp = dp[t_on[dv0:dv1]]
    y1_dp = dp[t_off[dv0:dv1]]

    start = t_on[dv0]
    stop  = t_off[dv1]
    ax1.plot(range(len(p[start:stop])), p[start:stop])
    ax1.scatter(x0, y0_p, color='red', label='start')
    ax1.scatter(x1, y1_p, color='blue', label='stop')
    ax1.set_ylabel('depth (m)')

    ax2.plot(range(len(dp[start:stop])), dp[start:stop])
    ax2.scatter(x0, y0_dp, color='red', label='start')
    ax2.scatter(x1, y1_dp, color='blue', label='stop')
    ax2.set_ylabel('depth (dm/t)')
    ax2.set_xlabel('sample')

    for ax in [ax1, ax2]:
        ax.legend(loc='upper right')
        ax.set_xlim([-50, len(dp[start:stop])+50])

    plt.show()


def test_contiguous_regions():
    import numpy

    x = numpy.arange(-10,10, 0.01)
    y= -x**2 + x + 10
    condition = y > 0
    start, stop = contiguous_regions(condition)
    # Assert that both

    assert (round(y[start[0]])==0) & (round(y[stop[0]])==0)

def contiguous_regions(condition):
    '''Finds contiguous True regions of the boolean array 'condition'.

    Args
    ----
    condition

    Returns
    -------
    an array with the start indices of the region
    an array with the stop indices of the region

    http://stackoverflow.com/a/4495197/943773
    '''
    import numpy

    # Find the indicies of changes in 'condition'
    d = numpy.diff(condition)
    idx, = d.nonzero()


    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = numpy.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = numpy.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)

    # We need to start things after the change in 'condition'. Therefore,
    # we'll shift the index by 1 to the right.
    start = idx[:,0] + 1
    # keep the stop ending just before the change in condition
    stop  = idx[:,1]

    # remove reversed or same start/stop index
    good_vals = (stop-start) > 0
    start = start[good_vals]
    stop = stop[good_vals]

    return start, stop


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
    t_on, t_off = contiguous_regions(condition)

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
