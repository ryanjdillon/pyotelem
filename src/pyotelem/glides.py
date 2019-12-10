
def get_stroke_freq(Ax, Az, fs_a, nperseg, peak_thresh, stroke_ratio=None):
    '''Determine stroke frequency to use as a cutoff for filtering

    Args
    ----
    Ax: numpy.ndarray, shape (n,)
        x-axis accelermeter data (longitudinal)
    Ay: numpy.ndarray, shape (n,)
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
    stroke_ratio: float

    Notes
    -----
    During all descents and ascents phases where mainly steady swimming occurs.
    When calculated for the whole dive it may be difficult to differentiate the
    peak at which stroking rate occurs as there are movements than
    only steady swimming

    Here the power spectra is calculated of the longitudinal and dorso-ventral
    accelerometer signals during descents and ascents to determine the dominant
    stroke frequency for each animal in each phase

    This numpy samples per f segment 512 and a sampling rate of fs_a.

    Output: S is the amount of power in each particular frequency (f)
    '''

    import numpy

    from . import dsp
    from . import utils
    from .plots import plotdsp

    # Axes to be used for determining `stroke_frq`
    stroke_axes = [(0,'x','dorsa-ventral', Ax),
                   (2,'z','lateral', Az)]

    # Lists for appending values from each axis
    cutoff_frqs   = list()
    stroke_frqs   = list()
    stroke_ratios = list()

    # Iterate over axes in `stroke_axes` list appending output to above lists
    for i, i_alph, name, data in stroke_axes:

        frqs, S, _, _ = dsp.calc_PSD_welch(data, fs_a, nperseg)

        # Find index positions of local maxima and minima in PSD
        delta = S.max()/1000
        max_ind, min_ind = dsp.simple_peakfinder(range(len(S)), S, delta)

        max0 = max_ind[0]

        # TODO hack fix, improve later
        try:
            min0 = min_ind[0]
        except:
            min0 = None
            stroke_ratio = 0.4

        stroke_frq = frqs[max0]

        # Prompt user for `cutoff_frq` value after inspecting PSD plot
        title = 'PSD - {} axis (n={}), {}'.format(i_alph, i, name)
        # Plot power spectrum against frequency distribution
        plotdsp.plot_welch_peaks(frqs, S, max_ind, title=title)

        # Get user input of cutoff frequency identified off plots
        cutoff_frq = utils.recursive_input('cutoff frequency', float)

        # Append values for axis to list
        cutoff_frqs.append(cutoff_frq)
        stroke_frqs.append(stroke_frq)
        stroke_ratios.append(stroke_ratio)

    # Average values for all axes
    cutoff_frq = float(numpy.mean(cutoff_frqs))
    stroke_frq = float(numpy.mean(stroke_frqs))

    # Handle exception of manual selection when `stroke_ratio == None`
    # TODO with fix
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
        Animal frame triaxial accelerometer matrix at sampling rate fs_a.

    fs_a: int
        Number of accelerometer samples per second

    J: float
        Frequency threshold for detecting a fluke stroke in m/s^2.  If J is not
        given, fluke strokes will not be located but the rotations signal (pry)
        will be computed.

    t_max: int
        Maximum duration allowable for a fluke stroke in seconds.  A fluke
        stroke is counted whenever there is a cyclic variation in the pitch
        deviation with peak-to-peak magnitude greater than +/-J and consistent
        with a fluke stroke duration of less than t_max seconds, e.g., for
        Mesoplodon choose t_max=4.

    Returns
    -------
    GL: 1-D ndarray
        Matrix containing the start time (first column) and end time (2nd
        column) of any glides (i.e., no zero crossings in t_max or more
        seconds). Times are in seconds.

    Note
    ----
    If no J or t_max is given, J=[], or t_max=[], GL returned as None
    '''
    import numpy

    from . import dsp

    # Check if input array is 1-D
    if A_g_hf.ndim > 1:
        raise IndexError('A_g_hf multidimensional: Glide index determination '
                         'requires 1-D acceleration array as input')

    # Convert t_max to number of samples
    n_max = t_max * fs_a

    # Find zero-crossing start/stops in pry(:,n), rotations around n axis.
    zc = dsp.findzc(A_g_hf, J, n_max/2)

    # find glides - any interval between zeros crossings greater than `t_max`
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
    '''Get start/stop indices of each `dur` length sub-glide for glides in GL

    Args
    ----
    dur: int
        Desired duration of glides
    GL: ndarray, (n, 2)
        Matrix containing the start time (first column) and end time
        (2nd column) of any glides.Times are in seconds.
    min_dur: int, default (bool False)
        Minimum number of seconds for sub-glide. Default value is `False`, which
        makes `min_dur` equal to `dur`, ignoring sub-glides smaller than `dur`.

    Attributes
    ----------
    gl_ind_diff: ndarray, (n,3)
        GL, with additional column of difference between the first two columns

    Returns
    -------
    SGL: ndarray, (n, 2)
        Matrix containing the start time (first column) and end time (2nd
        column) of the generated sub-glides. All glides must have duration equal
        to the given dur value.Times are in seconds.
    '''
    import numpy

    # Convert `dur` in seconds to duration in number of samples `ndur`
    ndur = dur * fs_a

    # If minimum duration not passed, set to `min_dur` to skip slices < `dur`
    if not min_dur:
        min_ndur = dur * fs_a
    else:
        min_ndur = min_dur * fs_a

    # `GL` plus column for total duration of glide, seconds
    gl_ind_diff = numpy.vstack((GL.T, GL[:, 1] - GL[:, 0])).T

    # Split all glides in `GL`
    SGL_started = False
    for i in range(len(GL)):
        gl_ndur = gl_ind_diff[i, 2]

        # Split into sub glides if longer than duration
        if abs(gl_ndur) > ndur:

            # Make list of index lengths to start of each sub-glide
            n_sgl     = int(gl_ndur//ndur)
            sgl_ndur  = numpy.ones(n_sgl)*ndur
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
                    # Concatenate 1D arrays together, shape (n,)
                    sgl_start_ind = numpy.hstack((sgl_start_ind, next_start_ind))
                    sgl_end_ind   = numpy.hstack((sgl_end_ind, next_end_ind))

    # Stack and transpose indices into shape (n, 2)
    SGL = numpy.vstack((sgl_start_ind, sgl_end_ind)).T

    # Filter out sub-glides that fall outside of sensor data indices
    SGL =  SGL[(SGL[:, 0] >= 0) & (SGL[:, 1] < n_samples)]

    # check that all sub-glides have a duration of `ndur` seconds
    sgl_ndur = SGL[:, 1] - SGL[:, 0]

    # If sub-glide `min_ndur` set, make sure all above `min_ndur`, below `ndur`
    if min_dur:
        assert numpy.all((sgl_ndur <= ndur) & (sgl_ndur >= min_ndur))
    # Else make sure all sample number durations equal to `ndur`
    else:
        assert numpy.all(sgl_ndur == ndur)

    # Create `data_sgl_mask`
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

    Returns
    -------
    sgls: pandas.DataFrame
        Sub-glide summary information defined by `SGL` start/stop indices

        *Columns*:

        * dive_phase
        * dive_id
        * dive_min_depth
        * dive_max_depth
        * dive_duration
        * start_idx
        * stop_idx
        * duration
        * mean_depth
        * total_depth_change
        * abs_depth_change
        * mean_speed
        * total_speed_change
        * mean_pitch
        * mean_sin_pitch
        * SD_pitch
        * mean_temp
        * mean_swdensity
        * mean_a
        * R2_speed_vs_time
        * SE_speed_vs_time
        * mean_pitch_circ
        * pitch_concentration
        * mean_roll_circ
        * roll_concentration
        * mean_heading_circ
        * heading_concentration

    '''
    import scipy.stats
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
        sin_pitch = numpy.sin(numpy.mean(pitch_lf_deg[start_idx:stop_idx]))
        sgls['mean_sin_pitch'][i] = sin_pitch

        # SD of pitch during the sub-glide
        # TODO just use original radian array, not deg
        #      make sure "original" is radians ;)
        sd_pitch = numpy.std(pitch_lf_deg[start_idx:stop_idx]) * 180 / numpy.pi
        sgls['SD_pitch'][i] = sd_pitch

        # mean temperature during the sub-glide
        sgls['mean_temp'][i] = numpy.mean(temperature[start_idx:stop_idx])

        # mean seawater density (kg/m^3) during the sub-glide
        sgls['mean_swdensity'][i] = numpy.mean(Dsw[start_idx:stop_idx])

        try:
            # Perform linear regression on sub-glide data subset
            xpoly = numpy.arange(start_idx, stop_idx).astype(int)
            ypoly = swim_speed[start_idx:stop_idx]

            # slope, intercept, r-value, p-value, standard error
            m, c, r, p, std_err = scipy.stats.linregress(xpoly, ypoly)

            # mean acceleration during the sub-glide
            sgls['mean_a'][i] = m

            # R2-value for the regression swim speed vs. time during the
            # sub-glide
            sgls['R2_speed_vs_time'][i] = r**2

            # SE of the gradient for the regression swim speed vs. time during
            # the sub-glide
            sgls['SE_speed_vs_time'][i] = std_err

        except:
            # mean acceleration during the sub-glide
            sgls['mean_a'][i] = numpy.nan

            # R2-value for the regression swim speed vs. time during the
            # sub-glide
            sgls['R2_speed_vs_time'][i] = numpy.nan

            # SE of the gradient for the regression swim speed vs. time during
            # the sub-glide
            sgls['SE_speed_vs_time'][i] = numpy.nan


        # TODO this does not take into account the positive depths, flipping
        # descents to ascents
        sgl_signs = numpy.sign(sgl_depth_diffs)
        if all(sgl_signs == -1):
            sgls['dive_phase'][i] = 'descent'
        elif all(sgl_signs == 1):
            sgls['dive_phase'][i] = 'ascent'
        elif any(sgl_signs == -1) & any(sgl_signs == 1):
            sgls['dive_phase'][i] = 'mixed'

        # Get indices of dive in which glide occurred
        dive_ind = numpy.where((dives['start_idx'] < start_idx) & \
                               (dives['stop_idx'] > stop_idx))[0]

        # TODO Hack way of pulling the information from first column of dives
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
        pitch_circmean = scipy.stats.circmean(pitch_lf[start_idx:stop_idx])
        sgls['mean_pitch_circ'][i] = pitch_circmean

        # Measure of concentration (r) of pitch during the sub-glide (i.e. 0
        # for random direction, 1 for unidirectional)
        pitch_circvar = scipy.stats.circvar(pitch_lf[start_idx:stop_idx])
        sgls['pitch_concentration'][i] = 1 - pitch_circvar

        # Mean roll (deg) calculated using circular statistics
        roll_circmean = scipy.stats.circmean(roll_lf[start_idx:stop_idx])
        sgls['mean_roll_circ'][i] = roll_circmean

        # Measure of concentration (r) of roll during the sub-glide
        roll_circvar = scipy.stats.circvar(roll_lf[start_idx:stop_idx])
        sgls['roll_concentration'][i] = 1 - roll_circvar

        # Mean heading (deg) calculated using circular statistics
        heading_circmean = scipy.stats.circmean(heading_lf[start_idx:stop_idx])
        sgls['mean_heading_circ'][i] = heading_circmean

        # Measure of concentration (r) of heading during the sub-glide
        heading_circvar = scipy.stats.circvar(heading_lf[start_idx:stop_idx])
        sgls['heading_concentration'][i] = 1 - heading_circvar

    return sgls


def calc_glide_ratios(dives, des, asc, glide_mask, depths, pitch_lf):
    '''Calculate summary information on glides during dive ascent/descents

    Args
    ----
    dives: (n,10)
        Numpy record array with summary information of dives in sensor data
    des: ndarray
        Boolean mask of descents over sensor data
    asc: ndarray
        Boolean mask of descents over sensor data
    glid_mask: ndarray
        Boolean mask of glides over sensor data
    depths: ndarray
        Depth values at each sensor sampling
    pitch_lf: ndarray
        Pitch in radians over the low frequency signals of acceleration

    Returns
    -------
    gl_ratio: pandas.DataFrame
        Dataframe of summary information of glides during dive descent/ascents

        *Columns*:

        * des_duration
        * des_gl_duration
        * des_gl_ratio
        * des_mean_pitch
        * des_rate
        * asc_duration
        * asc_gl_duration
        * asc_gl_ratio
        * asc_mean_pitch
        * asc_rate
    '''
    import numpy
    import pandas

    # Create empty `pandas.DataFrame` for storing data, init with `nan`s
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
        # total duration of the descent phase (s)
        gl_ratio['des_duration'][i] = len(des_ind)

        # total glide duration during the descent phase (s)
        des_glides = numpy.where(glide_mask[des_ind])[0]
        gl_ratio['des_gl_duration'][i] = len(des_glides)

        if len(des_ind) == 0:
            gl_ratio['des_gl_ratio'][i] = 0
            gl_ratio['des_rate'][i] = 0
        else:
            # glide ratio during the descent phase
            des_gl_ratio = gl_ratio['des_gl_duration'][i] / len(des_ind)
            gl_ratio['des_gl_ratio'][i] = des_gl_ratio

            # descent rate (m/sample)
            max_depth_des = depths[des_ind].max()
            gl_ratio['des_rate'][i] = max_depth_des / len(des_ind)

        # mean pitch during the descent phase(degrees)
        des_mean_pitch = numpy.mean(numpy.rad2deg(pitch_lf[des_ind]))
        gl_ratio['des_mean_pitch'][i] = des_mean_pitch

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
            asc_gl_ratio = gl_ratio['asc_gl_duration'][i] / len(asc_ind)
            gl_ratio['asc_gl_ratio'][i] = asc_gl_ratio

            # ascent rate (m/sample)
            max_depth_asc = depths[asc_ind].max()
            gl_ratio['asc_rate'][i] = max_depth_asc / len(asc_ind)

        # mean pitch during the ascent phase(degrees)
        asc_mean_pitch = numpy.mean(numpy.rad2deg(pitch_lf[asc_ind]))
        gl_ratio['asc_mean_pitch'][i] = asc_mean_pitch

    return gl_ratio
