
def get_stroke_freq(Ax, Az, fs_a, nperseg, peak_thresh, auto_select=False,
        stroke_ratio=None):
    '''Determine stroke frequency to use as a cutoff for filtering

    Args
    ----
    cfg: dict
        parameters for configuration of body condition analysis
    Ax: numpy.ndarray, shape (n,)
        x-axis accelermeter data (longitudinal)
    ay: numpy.ndarray, shape (n,)
        x-axis accelermeter data (lateral)
    Az: numpy.ndarray, shape (n,)
        z-axis accelermeter data (dorso-ventral)
    fs_a: float
        sampling frequency (i.e. number of samples per second)
    nperseg: int
        length of each segment (i.e. number of samples per frq. band in PSD
        calculation. Default to 512 (scipy.signal.welch() default is 256)
    peak_thresh: float
        PSD power level threshold. Only peaks over this threshold are returned.

    Returns
    -------
    cutoff_frq: float
        cutoff frequency of signal (Hz) to be used for low/high-pass filtering
    stroke_frq: float
        frequency of dominant wavelength in signal

    Notes
    -----
    During all descents and ascents phases where mainly steady swimming occurs.
    When calculated for the whole dive it may be difficult to differenciate the
    peak at which stroking rate occurs as there is other kind of movements than
    only steady swimming

    Here the power spectra is calculated of the longitudinal and dorso-ventral
    accelerometer signals during descents and ascents to determine the dominant
    stroke frequency for each animal in each phase

    This numpyer samples per f segment 512 and a sampling rate of fs_a.

    Output: S is the amount of power in each particular frequency (f)
    '''

    #def plot_PSD_peaks(f, S, max_ind, min_ind, stroke_ratio_idx):
    #    import matplotlib.pyplot as plt

    #    peak_x = f[max_ind]
    #    peak_y = S[max_ind]
    #    low_x = f[min_ind]
    #    low_y = S[min_ind]
    #    plt.plot(f, S)
    #    plt.scatter(peak_x, peak_y)
    #    plt.scatter(low_x, low_y)
    #    plt.scatter(f[stroke_ratio_idx], S[stroke_ratio_idx])
    #    plt.show()

    #    return None

    #def auto_cutoff(f, S):
    #    import utils
    #    import utils_signal

    #    # Find index positions of local maxima and minima in PSD
    #    delta = S.max()/1000
    #    max_ind, min_ind = utils_signal.simple_peakfinder(range(len(S)), S,
    #                                                      delta)
    #    max0 = max_ind[0]
    #    min0 = min_ind[0]

    #    # Get percentage after max peak to use as filter cutoff
    #    stroke_ratio = 100*S[min0]/S[max0]

    #    # Find nearest PSD value between first peak and trough that matches
    #    nearest_S = utils.nearest(S[max0:min0], stroke_ratio*S[max0])

    #    # Get index position of this point and associated frequency for cuttoff
    #    stroke_ratio_idx = max0 + numpy.where(S==nearest_S)[0][0]
    #    cutoff = f[stroke_ratio_idx]

    #    return cutoff, stroke_ratio

    import matplotlib.pyplot as plt
    import numpy

    import utils
    import utils_plot
    import utils_signal

    # Axes to be used for determining `stroke_frq`
    stroke_axes = [(0,'x','dorsa-ventral', Ax),
                   (2,'z','lateral', Az)]

    # Lists for appending values from each axis
    cutoff_frqs   = list()
    stroke_frqs   = list()
    stroke_ratios = list()

    # Iterate over axes in `stroke_axes` list appending output to above lists
    for i, i_alph, name, data in stroke_axes:

        frqs, S, _, _ = utils_signal.calc_PSD_welch(data, fs_a, nperseg)

        # Find index positions of local maxima and minima in PSD
        delta = S.max()/1000
        max_ind, min_ind = utils_signal.simple_peakfinder(range(len(S)), S,
                                                          delta)
        #peak_loc = peak_loc[S[peak_loc] > peak_thresh]

        max0 = max_ind[0]

        # TODO hack fix, improve later
        try:
            min0 = min_ind[0]
        except:
            min0 = None
            stroke_ratio = 0.4

        stroke_frq = frqs[max0]

        # Auto calculate, else promter user for value after inspecting PSD plot
        if auto_select == True:
            if min0 and not stroke_ratio:
                # Get percentage after max peak to use as filter cutoff
                stroke_ratio = 100*S[min0]/S[max0]

            # Find nearest PSD value between first peak and trough that matches
            nearest_S = utils.nearest(S[max0:], stroke_ratio*S[max0])

            # Get index position of this point and associated frequency for cuttoff
            stroke_ratio_idx = max0 + numpy.where(S==nearest_S)[0][0]

            cutoff_frq   = frqs[stroke_ratio_idx]

        elif auto_select == False:
            title = 'PSD - {} axis (n={}), {}'.format(i_alph, i, name)
            # Plot power specturm against frequency distribution
            utils_plot.plot_welch_peaks(frqs, S, max_ind, title=title)

            # Get user input of cutoff frequency identified off plots
            cutoff_frq = utils.recursive_input('cutoff frequency', float)

        # Append values for axis to list
        cutoff_frqs.append(cutoff_frq)
        stroke_frqs.append(stroke_frq)
        stroke_ratios.append(stroke_ratio)

    # Average values for all axes and update config dictionary
    cutoff_frq   = float(numpy.mean(cutoff_frqs))
    stroke_frq   = float(numpy.mean(stroke_frqs))

    # Handle exception of manual selection when `stroke_ratio == None`
    try:
        stroke_ratio = float(numpy.mean(stroke_ratios))
    except:
        stroke_ratio = None

    return cutoff_frq, stroke_frq, stroke_ratio


def get_stroke_glide_indices(A_g_hf, fs_a, J, t_max):
    '''Get stroke and glide indices from high-pass accelerometer data

    Args
    ----
    A_g_hf: 1-D ndarray
       animal frame triaxial accelerometer matrix at sampling rate fs_a.

    fs_a: int
        number of acclerometer samples per second

    J: float
        frequency threshold for detecting a fluke stroke in m/s2.  If J is not
        given, fluke strokes will not be located but the rotations signal (pry)
        will be computed.

    t_max: int
        maximum duration allowable for a fluke stroke in seconds.  A fluke
        stroke is counted whenever there is a cyclic variation in the pitch
        deviation with peak-to-peak magnitude greater than +/-J and consistent
        with a fluke stroke duration of less than t_max seconds, e.g., for
        Mesoplodon choose t_max=4.

    Returns
    -------
    GL: 1-D ndarray
        matrix containing the start time (first column) and end time (2nd
        column) of any glides (i.e., no zero crossings in t_max or more
        seconds).Times are in seconds.

    Note
    ----
    If no J or t_max is given, J=[], or t_max=[], GL returned as None

    `K`   changed to `zc`
    `kk`  changed to `col`
    `glk` changed to `gl_ind`

    '''
    import numpy

    import utils_signal

    # Check if input array is 1-D
    if A_g_hf.ndim > 1:
        raise IndexError('A_g_hf multidimensional: Glide index determination '
                         'requires 1-D acceleration array as input')

    # Convert t_max to number of samples
    n_max = t_max * fs_a

    # Find zero-crossing start/stops in pry(:,n), rotations around n axis.
    zc = utils_signal.findzc(A_g_hf, J, n_max/2)

    # find glides - any interval between zeros crossings greater than tmax
    ind = numpy.where(zc[1:, 0] - zc[0:-1, 1] > n_max)[0]
    gl_ind = numpy.vstack([zc[ind, 0] - 1, zc[ind + 1, 1] + 1]).T

    # Compute mean index position of glide, Only include sections with jerk < J
    gl_mean_idx = numpy.round(numpy.mean(gl_ind, 1)).astype(int)
    gl_ind = numpy.round(gl_ind).astype(int)

    for i in range(len(gl_mean_idx)):
        col = range(gl_mean_idx[i], gl_ind[i, 0], - 1)
        test = numpy.where(numpy.isnan(A_g_hf[col]))[0]
        if test.size != 0:
            gl_mean_idx[i]   = numpy.nan
            gl_ind[i,0] = numpy.nan
            gl_ind[i,1] = numpy.nan
        else:
            over_J1 = numpy.where(abs(A_g_hf[col]) >= J)[0][0]

            gl_ind[i,0] = gl_mean_idx[i] - over_J1 + 1

            col = range(gl_mean_idx[i], gl_ind[i, 1])

            over_J2 = numpy.where(abs(A_g_hf[col]) >= J)[0][0]

            gl_ind[i,1] = gl_mean_idx[i] + over_J2 - 1

    GL = gl_ind
    GL = GL[numpy.where(GL[:, 1] - GL[:, 0] > n_max / 2)[0], :]

    return GL



def split_glides(n_samples, dur, fs_a, GL, min_dur=None):
    '''Get start/stop indices of each `dur` lenth sub-glide for glides in GL

    Args
    ----
    dur: int
        desired duration of glides

    GL: numpy.ndarray, shape(n, 2)
        matrix containing the start time (first column) and end time
        (2nd column) of any glides.Times are in seconds.

    min_dur: int, default (bool False)
        minimum number of seconds for subglide. Default value is `False`, which
        makes `min_dur` equal to `dur`, ignoring sub-glides smaller than `dur`.

    Attributes
    ----------
    gl_ind_diff: numpy.ndarray, shape(n,3)
        GL, with aditional column of diffence between the first two columns

    Returns
    -------
    SGL: numpy.ndarray, shape(n, 2)
        matrix containing the start time (first column) and end time(2nd
        column) of the generated subglides.All glides must have duration
        equal to the given dur value.Times are in seconds.

    Note
    ----
    `SUM`         removed
    `ngl`         renamed `n_sgl`
    `glinf`       renamed `gl_ind_diff`
    `SGL`         renamed `SGL`
    `STARTGL`     renamed `sgl_start_ind`
    `ENDGL`       renamed `sgl_end_ind`
    `startglide1` renamed `next_SGL`
    `endglide1`   renamed `next_SGL`
    `v`           renamed `sgl_start`

    Lucia Martina Martin Lopez (May 2016)
    lmml2@st-andrews.ac.uk
    '''
    import numpy

    # Convert `dur` in seconds to number of samples
    ndur = dur * fs_a

    # If minimum duration not passed, set to `min_dur` to skip slices < `dur`
    if not min_dur:
        min_ndur = dur * fs_a
    else:
        min_ndur = min_dur * fs_a

    # GL plus column for total duration of glide, seconds
    gl_ind_diff = numpy.vstack((GL.T, GL[:, 1] - GL[:, 0])).T

    # Split all glides in GL
    SGL_started = False
    for i in range(len(GL)):
        gl_ndur = gl_ind_diff[i, 2]

        # Split into sub glides if longer than duration
        if abs(gl_ndur) > ndur:

            # Make list of index lengths to start of each sub-glide
            n_sgl     = int(gl_ndur//ndur)
            sgl_ndur   = numpy.ones(n_sgl)*ndur
            sgl_start = numpy.arange(n_sgl)*(ndur+1)

            # Add remainder as a sub-glide, skips if `min_ndur` not passed
            if (gl_ndur%ndur > min_ndur):
                last_ndur = numpy.floor(gl_ndur%ndur)
                sgl_ndur  = numpy.hstack([sgl_ndur, last_ndur])

                last_start = (len(sgl_start)*ndur) + ndur
                sgl_start  = numpy.hstack([sgl_start, last_start])

            # Get start and end index positions for each sub-glide
            for k in range(len(sgl_start)):

                # starting at original glide start...
                # sgl_start_ind: add index increments of ndur+1 for next start idx
                next_start_ind = (gl_ind_diff[i, 0] + sgl_start[k]).astype(int)

                # end_glide: add `ndur` to that to get ending idx
                next_end_ind = (next_start_ind + sgl_ndur[k]).astype(int)

                # If first iteration, set equal to first set of indices
                if SGL_started == False:
                    sgl_start_ind = next_start_ind
                    sgl_end_ind   = next_end_ind
                    SGL_started = True
                else:
                    # Concat 1D arrays together, shape (n,)
                    sgl_start_ind = numpy.hstack((sgl_start_ind, next_start_ind))
                    sgl_end_ind   = numpy.hstack((sgl_end_ind, next_end_ind))

    # Stack and transpose indices into shape (n, 2)
    SGL = numpy.vstack((sgl_start_ind, sgl_end_ind)).T

    # Filter out subglides that fall outside of sensor data indices
    SGL =  SGL[(SGL[:, 0] >= 0) & (SGL[:, 1] < n_samples)]

    # check that all subglides have a duration of `ndur` seconds
    sgl_ndur = SGL[:, 1] - SGL[:, 0]

    # If subglide `min_ndur` set, make sure all above `min_ndur`, below `ndur`
    if min_dur:
        assert numpy.all((sgl_ndur <= ndur) & (sgl_ndur >= min_ndur))
    # Else make sure all nduration equal to `ndur`
    else:
        assert numpy.all(sgl_ndur == ndur)

    # Create data_sgl_mask
    data_sgl_mask = numpy.zeros(n_samples, dtype=bool)
    for start, stop in SGL.astype(int):
        data_sgl_mask[start:stop] = True

    return SGL, data_sgl_mask


def calc_glide_des_asc(depths, pitch_lf, roll_lf, heading_lf, swim_speed,
        dives, SGL, pitch_lf_deg, temperature, Dsw):
    '''Calculate glide ascent and descent summary data

    Args
    ----
    SGL: numpy.ndarray, shape (n,2)
        start and end index positions for sub-glides
    '''
    import astropy.stats
    import numpy
    import pandas

    keys = [
            'dive_phase',
            'dive_id',
            'dive_min_depth',
            'dive_max_depth',
            'dive_duration',
            'start_idx',
            'stop_idx',
            'duration',
            'mean_depth',
            'total_depth_change',
            'abs_depth_change',
            'mean_speed',
            'total_speed_change',
            'mean_pitch',
            'mean_sin_pitch',
            'SD_pitch',
            'mean_temp',
            'mean_swdensity',
            'mean_a',
            'R2_speed_vs_time',
            'SE_speed_vs_time',
            'mean_pitch_circ',
            'pitch_concentration',
            'mean_roll_circ',
            'roll_concentration',
            'mean_heading_circ',
            'heading_concentration',
            ]

    # Create empty pandas dataframe for summary values
    sgls = pandas.DataFrame(index=range(len(SGL)), columns=keys)

    for i in range(len(SGL)):
        start_idx = SGL[i,0]
        stop_idx = SGL[i,1]

        # sub-glide start point in seconds
        sgls['start_idx'][i] = start_idx

        # sub-glide end point in seconds
        sgls['stop_idx'][i] = stop_idx

        # sub-glide duration
        sgls['duration'][i] = SGL[i,1] - SGL[i,0]

        # mean depth(m)during sub-glide
        sgls['mean_depth'][i] = numpy.mean(depths[start_idx:stop_idx])

        # total depth(m)change during sub-glide
        sgl_depth_diffs = numpy.diff(depths[start_idx:stop_idx])
        sgls['total_depth_change'][i] = numpy.sum(abs(sgl_depth_diffs))

        # depth(m)change from start to end of sub-glide
        sgls['abs_depth_change'][i] = abs(depths[start_idx] - depths[stop_idx])

        # mean swim speed during the sub-glide, only given if pitch>30 degrees
        sgls['mean_speed'][i] = numpy.nanmean(swim_speed[start_idx:stop_idx])

        # total speed change during sub-glide
        sgl_speed_diffs = numpy.diff(swim_speed[start_idx:stop_idx])
        sgls['total_speed_change'][i] = numpy.nansum(abs(sgl_speed_diffs))

        # mean pitch during the sub-glide
        sgls['mean_pitch'][i] = numpy.mean(pitch_lf_deg[start_idx:stop_idx])

        # mean sin pitch during the sub-glide
        sgls['mean_sin_pitch'][i] = numpy.sin(numpy.mean(pitch_lf_deg[start_idx:stop_idx]))

        # SD of pitch during the sub-glide
        # TODO just use original radian array, not deg
        #      make sure "original" is radians ;)
        sgls['SD_pitch'][i] = numpy.std(pitch_lf_deg[start_idx:stop_idx]) * 180 / numpy.pi

        # mean temperature during the sub-glide
        sgls['mean_temp'][i] = numpy.mean(temperature[start_idx:stop_idx])

        # mean seawater density (kg/m^3) during the sub-glide
        sgls['mean_swdensity'][i] = numpy.mean(Dsw[start_idx:stop_idx])

        try:
            # Perform linear regression on subglide data subset
            xpoly = numpy.arange(start_idx, stop_idx).astype(int)
            ypoly = swim_speed[start_idx:stop_idx]

            import scipy.stats

            # slope, intercept, r-value, p-value, standard error
            m, c, r, p, std_err = scipy.stats.linregress(xpoly, ypoly)

            # mean acceleration during the sub-glide
            sgls['mean_a'][i] = m

            # R2-value for the regression swim speed vs. time during the sub-glide
            sgls['R2_speed_vs_time'][i] = r**2

            # SE of the gradient for the regression swim speed vs. time during the
            # sub-glide
            sgls['SE_speed_vs_time'][i] = std_err

        except:

            # mean acceleration during the sub-glide
            sgls['mean_a'][i] = numpy.nan

            # R2-value for the regression swim speed vs. time during the sub-glide
            sgls['R2_speed_vs_time'][i] = numpy.nan

            # SE of the gradient for the regression swim speed vs. time during the
            # sub-glide
            sgls['SE_speed_vs_time'][i] = numpy.nan


        # TODO remove?
        # Dive phase:0 bottom, -1 descent, 1 ascent, NaN not dive phase
        #sumphase = sum(phase[start_idx:stop_idx])
        #sp = numpy.nan
        #if sumphase < 0:
        #    sp = -1
        #elif sumphase == 0:
        #    sp = 0
        #elif sumphase > 0:
        #    sp = 1
        sgl_signs = numpy.sign(sgl_depth_diffs)
        if all(sgl_signs == -1):
            sgls['dive_phase'][i] = 'descent'
        elif all(sgl_signs == 1):
            sgls['dive_phase'][i] = 'ascent'
        elif any(sgl_signs == -1) & any(sgl_signs == 1):
            sgls['dive_phase'][i] = 'mixed'

        # Get indices of dive in which glide occured
        dive_ind = numpy.where((dives['start_idx'] < start_idx) & \
                               (dives['stop_idx'] > stop_idx))[0]

        # TODO Hokey way of pulling the information from first column of dives
        if dive_ind.size == 0:
            dive_inf = pandas.DataFrame.copy(dives.iloc[0])
            dive_inf[:] = numpy.nan
        else:
            dive_inf = dives.iloc[dive_ind].iloc[0]

        # Dive number in which the sub-glide recorded
        sgls['dive_id'][i] = dive_inf['dive_id']

        # Minimum dive depth (m) of the dive
        sgls['dive_min_depth'][i] = dive_inf['depth_min']

        # Maximum dive depth (m) of the dive
        sgls['dive_max_depth'][i] = dive_inf['depth_max']

        # Dive duration (s) of the dive
        sgls['dive_duration'][i] = dive_inf['dive_dur']

        # Mean pitch(deg) calculated using circular statistics
        sgls['mean_pitch_circ'][i] = astropy.stats.circmean(pitch_lf[start_idx:stop_idx])

        # Measure of concentration (r) of pitch during the sub-glide (i.e. 0 for
        # random direction, 1 for unidirectional)
        sgls['pitch_concentration'][i] = 1 - astropy.stats.circvar(pitch_lf[start_idx:stop_idx])

        # Mean roll (deg) calculated using circular statistics
        sgls['mean_roll_circ'][i] = astropy.stats.circmean(roll_lf[start_idx:stop_idx])

        # Measure of concentration (r) of roll during the sub-glide
        sgls['roll_concentration'][i] = 1 - astropy.stats.circvar(roll_lf[start_idx:stop_idx])

        # Mean heading (deg) calculated using circular statistics
        sgls['mean_heading_circ'][i] = astropy.stats.circmean(heading_lf[start_idx:stop_idx])

        # Measure of concentration (r) of heading during the sub-glide
        sgls['heading_concentration'][i] = 1 - astropy.stats.circvar(heading_lf[start_idx:stop_idx])

    return sgls


def calc_glide_ratios(dives, des, asc, glide_mask, depths, pitch_lf):
    import numpy
    import pandas

    # Create empty pandas dataframe for storing data, init'd with nans
    cols = ['des_duration',
            'des_gl_duration',
            'des_gl_ratio',
            'des_mean_pitch',
            'des_rate',
            'asc_duration',
            'asc_gl_duration',
            'asc_gl_ratio',
            'asc_mean_pitch',
            'asc_rate',]

    gl_ratio = pandas.DataFrame(index=range(len(dives)), columns=cols)

    # For each dive with start/stop indices in `dives`
    for i in range(len(dives)):
        # Get indices for descent and ascent phase of dive `i`
        start_idx = dives['start_idx'][i]
        stop_idx  = dives['stop_idx'][i]
        des_ind = numpy.where(des[start_idx:stop_idx])[0]
        asc_ind = numpy.where(asc[start_idx:stop_idx])[0]

        # DESCENT
        # total duration of the descet phase (s)
        gl_ratio['des_duration'][i] = len(des_ind)

        # total glide duration during the descet phase (s)
        des_glides = numpy.where(glide_mask[des_ind])[0]
        gl_ratio['des_gl_duration'][i] = len(des_glides)

        if len(des_ind) == 0:
            gl_ratio['des_gl_ratio'][i] = 0
            gl_ratio['des_rate'][i] = 0
        else:
            # glide ratio during the descent phase
            gl_ratio['des_gl_ratio'][i] = gl_ratio['des_gl_duration'][i] / len(des_ind)

            # descent rate (m/s) #TODO changed to (m/sample)
            max_depth_des = depths[des_ind].max()
            gl_ratio['des_rate'][i] = max_depth_des / len(des_ind)

        # mean pitch during the descent phase(degrees)
        gl_ratio['des_mean_pitch'][i] = numpy.mean(numpy.rad2deg(pitch_lf[des_ind]))


        # ASCENT
        # total duration of the ascent phase (s)
        gl_ratio['asc_duration'][i] = len(asc_ind)

        # total glide duration during the ascent phase (s)
        asc_glides = numpy.where(glide_mask[asc_ind])[0]
        gl_ratio['asc_gl_duration'][i] = len(asc_glides)

        if len(asc_ind) == 0:
            gl_ratio['asc_gl_ratio'][i] = 0
            gl_ratio['asc_rate'][i] = 0
        else:
            # glide ratio during the ascent phase
            gl_ratio['asc_gl_ratio'][i] = gl_ratio['asc_gl_duration'][i] / len(asc_ind)

            # ascent rate (m/s) #TODO changed to (m/sample)
            max_depth_asc = depths[asc_ind].max()
            gl_ratio['asc_rate'][i] = max_depth_asc / len(asc_ind)

        # mean pitch during the ascent phase(degrees)
        gl_ratio['asc_mean_pitch'][i] = numpy.mean(numpy.rad2deg(pitch_lf[asc_ind]))

    return gl_ratio
