
def get_gl_mask(depths, fs_a, GL):
    '''Get phase of subglides, depth at glide, duration, times'''
    import numpy

    import utils
    t = numpy.arange(len(depths))/fs_a

    #TODO make GL created as index numbers

    # Accelerometer (orginally in addition to magnetometer method)
    gl_ndur             = GL[:,1] - GL[:,0]
    gl_T               = numpy.vstack([GL[:,0], gl_ndur]).T
    gl_mask, _         = utils.event_on(gl_T, t)

    #for start, end in (GL/fs_a).round().astype(int):
    #    gl_mask[start:end] = True

    return gl_mask


def get_stroke_freq(x, fs_a, f, nperseg, peak_thresh, auto_select=False,
        default_cutoff=True):
    '''Determine stroke frequency to use as a cutoff for filtering

    Args
    ----
    cfg: dict
        parameters for configuration of body condition analysis
    x: numpy.ndarray, shape (n,3)
        tri-axial accelerometer data
    fs_a: float
        sampling frequency (i.e. number of samples per second)
    nperseg: int
        length of each segment (i.e. number of samples per frq. band in PSD
        calculation. Default to 512 (scipy.signal.welch() default is 256)
    peak_thresh: float
        PSD power level threshold. Only peaks over this threshold are returned.

    Returns
    -------
    cutoff: float
        cutoff frequency of signal (Hz) to be used for low/high-pass filtering
    stroke_f: float
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

    def automatic_freq(x, fs_a, nperseg, peak_thresh):
        '''Find peak of FFT PSD and set cutoff and stroke freq by this'''
        import utils_signal

        f_welch, S, _, _ = utils_signal.calc_PSD_welch(x, fs_a, nperseg)

        smooth_S = utils_signal.runmean(S, 10)

        # TODO check that arguments are same as dtag2 peakfinder
        peak_loc, peak_mag = utils_signal.peakfinder(S, sel=None, thresh=peak_thresh)

        peak_mag = peak_mag - smooth_S[peak_loc]
        peak_idx = peak_loc[peak_mag == max(peak_mag)]

        min_f    = numpy.min(S[:peak_idx])
        idx_f    = numpy.argmin(S[:peak_idx])

        cutoff         = f_welch[idx_f]
        stroke_f       = f_welch[peak_idx]

        return cutoff, stroke_f


    def manual_freq(x, fs_a, nperseg, peak_thresh, title):
        '''Manually look at plot, then enter cutoff frequency for x,y,z'''
        import matplotlib.pyplot as plt

        import utils
        import utils_plot
        import utils_signal

        f_welch, S, _, _ = utils_signal.calc_PSD_welch(x, fs_a, nperseg)

        # TODO this is a number that just works, better solution? put in cfg?
        sel = 8/fs_a
        peak_loc, peak_mag = utils_signal.peakfinder(S, sel=sel, peak_thresh=peak_thresh)
        peak_idx = peak_loc[peak_mag == max(peak_mag)]

        # Plot power specturm against frequency distribution
        utils_plot.plot_welch_peaks(f_welch, S, peak_loc, title=title)

        # Get user input of cutoff frequency identified off plots
        cutoff = utils.recursive_input('cutoff frequency', float)
        stroke_f = f_welch[peak_idx]

        return cutoff, stroke_f

    import numpy

    # Temporary error checking to be sure `auto_select` place holder not used
    if auto_select == True:
        raise SystemError('The automatic `stroke_f` selection method needs '
                          'further testing--implementation not active')

    # Axes to be used for determining `stroke_f`
    stroke_axes = [(0,'x','dorsa-ventral'),
                   (2,'z','lateral')]

    # Lists for appending values from each axis
    cutoff_all   = list()
    stroke_f_all = list()
    f_all        = list()

    # Iterate over axes in `stroke_axes` list appending output to above lists
    for i, i_alph, name in stroke_axes:
        # Auto calculate, else promter user for value after inspecting PSD plot
        if auto_select == True:
            cutoff, stroke_f = automatic_freq(x, fs_a, nperseg, peak_thresh)
        elif auto_select == False:
            title = 'PSD - {} axis (n={}), {}'.format(i_alph, i, name)
            cutoff, stroke_f =  manual_freq(x[:,i], fs_a, nperseg, peak_thresh, title)

        # Append values for axis to list
        cutoff_all.append(cutoff)
        stroke_f_all.append(stroke_f)

    # Average values for all axes and update config dictionary
    stroke_f = float(numpy.mean(stroke_f_all))
    cutoff   = float(numpy.mean(cutoff_all))

    if default_cutoff:
        # leave cfg['f'] as value set in load_config(), do nothing
        pass
    else:
        # Calculate `f` as per line 199 in the .m code
        f = cutoff/stroke_f

    return cutoff, stroke_f, f


def get_stroke_glide_indices(A_g_hf_n, fs_a, n, J, t_max):
    '''Get stroke and glide indices from high-pass accelerometer data

    Args
    ----
    A_g_hf_n: 1-D ndarray
       whale frame triaxial accelerometer matrix at sampling rate fs_a.

    fs_a: int
        number of acclerometer samples per second

    n: int
        fundamental axis of the acceleration signal.
        1 for accelerations along the x axis, longitudinal axis.
        2 for accelerations along the y axis, lateral axis.
        3 for accelerations along the z axis, dorso-ventral axis.

    J: float
        magnitude threshold for detecting a fluke stroke in m/s2.  If J is not
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
    if A_g_hf_n.ndim > 1:
        raise IndexError('A_g_hf_n multidimensional: Glide index determination '
                         'requires 1-D acceleration array as input')

    # Convert t_max to number of samples
    n_max = t_max * fs_a

    # Find zero-crossing start/stops in pry(:,n), rotations around n axis.
    zc = utils_signal.findzc(A_g_hf_n, J, n_max/2)

    # find glides - any interval between zeros crossings greater than tmax
    ind = numpy.where(zc[1:, 0] - zc[0:-1, 1] > n_max)[0]
    gl_ind = numpy.vstack([zc[ind, 0] - 1, zc[ind + 1, 1] + 1]).T

    # Compute mean index position of glide, Only include sections with jerk < J
    gl_c = numpy.round(numpy.mean(gl_ind, 1)).astype(int)
    gl_ind = numpy.round(gl_ind).astype(int)

    for i in range(len(gl_c)):
        col = range(gl_c[i], gl_ind[i, 0], - 1)
        test = numpy.where(numpy.isnan(A_g_hf_n[col]))[0]
        if test.size != 0:
            gl_c[i]   = numpy.nan
            gl_ind[i,0] = numpy.nan
            gl_ind[i,1] = numpy.nan
        else:
            over_J1 = numpy.where(abs(A_g_hf_n[col]) >= J)[0][0]

            gl_ind[i,0] = gl_c[i] - over_J1 + 1

            col = range(gl_c[i], gl_ind[i, 1])

            over_J2 = numpy.where(abs(A_g_hf_n[col]) >= J)[0][0]

            gl_ind[i,1] = gl_c[i] + over_J2 - 1

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
    sgl_ind: numpy.ndarray, shape(n, 2)
        matrix containing the start time (first column) and end time(2nd
        column) of the generated subglides.All glides must have duration
        equal to the given dur value.Times are in seconds.

    Note
    ----
    `SUM`         removed
    `ngl`         renamed `n_sgl`
    `glinf`       renamed `gl_ind_diff`
    `SGL`         renamed `sgl_ind`
    `STARTGL`     renamed `sgl_start_ind`
    `ENDGL`       renamed `sgl_end_ind`
    `startglide1` renamed `next_sgl_ind`
    `endglide1`   renamed `next_sgl_ind`
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
    sgl_ind_started = False
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

                # TODO round these to ints? or they are times...
                # starting at original glide start...
                # sgl_start_ind: add index increments of ndur+1 for next start idx
                next_start_ind = gl_ind_diff[i, 0] + sgl_start[k]

                # end_glide: add `ndur` to that to get ending idx
                next_end_ind = next_start_ind + sgl_ndur[k]

                # If first iteration, set equal to first set of indices
                if sgl_ind_started == False:
                    sgl_start_ind = next_start_ind
                    sgl_end_ind   = next_end_ind
                    sgl_ind_started = True
                else:
                    # Concat 1D arrays together, shape (n,)
                    sgl_start_ind = numpy.hstack((sgl_start_ind, next_start_ind))
                    sgl_end_ind   = numpy.hstack((sgl_end_ind, next_end_ind))

    # Stack and transpose indices into shape (n, 2)
    sgl_ind = numpy.vstack((sgl_start_ind, sgl_end_ind)).T

    # check that all subglides have a duration of `ndur` seconds
    sgl_ndur = sgl_ind[:, 1] - sgl_ind[:, 0]

    # If subglide `min_ndur` set, make sure all above `min_ndur`, below `ndur`
    if min_dur:
        assert numpy.all((sgl_ndur <= ndur) & (sgl_ndur >= min_ndur))
    # Else make sure all nduration equal to `ndur`
    else:
        assert numpy.all(sgl_ndur == ndur)

    # Create sgl_mask
    sgl_mask = numpy.zeros(n_samples, dtype=bool)
    for start, stop in sgl_ind:
        sgl_mask[start:stop] = True

    return sgl_ind, sgl_mask


def calc_glide_des_asc( depths, fs_a, pitch_lf, roll_lf, heading_lf, swim_speed,
        D, phase, sgl_ind, pitch_lf_deg, temperature, Dsw):
    '''Calculate glide ascent and descent summary data

    Args
    ----
    sgl_ind: numpy.ndarray, shape (n,2)
        start and end index positions for sub-glides
    '''
    import astropy.stats
    import numpy
    import pandas

    keys = ['start_idx',
            'stop_idx',
            'duration',
            'mean_depth',
            'depth_change',
            'mean_swimspeed',
            'mean_pitch',
            'mean_sin_pitch',
            'SD_pitch',
            'mean_temp',
            'mean_swdensity',
            'mean_acceleration',
            'R2_speed_vs_time',
            'SE_speed_vs_time',
            'dive_phase',
            'dive_number',
            'dive_max_depth',
            'dive_duration',
            'mean_pitch_circ',
            'pitch_concentration',
            'mean_roll_circ',
            'roll_concentration',
            'mean_heading_circ',
            'heading_concentration',]

    # Create empty pandas dataframe for summary values
    glide = pandas.DataFrame(index=range(len(sgl_ind)), columns=keys)

    for i in range(len(sgl_ind)):
        idx_start = int(round(sgl_ind[i,0]))
        idx_end = int(round(sgl_ind[i,1]))

        # sub-glide start point in seconds
        glide['start_idx'][i] = sgl_ind[i,0]

        # sub-glide end point in seconds
        glide['stop_idx'][i] = sgl_ind[i,1]

        # sub-glide duration
        glide['duration'][i] = sgl_ind[i,1] - sgl_ind[i,0]

        # mean depth(m)during sub-glide
        glide['mean_depth'][i] = numpy.mean(depths[idx_start:idx_end])

        # total depth(m)change during sub-glide
        glide['depth_change'][i] = abs(depths[idx_start] - depths[idx_end])

        # mean swim speed during the sub-glide, only given if pitch>30 degrees
        glide['mean_swimspeed'][i] = numpy.mean(swim_speed[idx_start:idx_end])

        # mean pitch during the sub-glide
        glide['mean_pitch'][i] = numpy.mean(pitch_lf_deg[idx_start:idx_end])

        # mean sin pitch during the sub-glide
        glide['mean_sin_pitch'][i] = numpy.sin(numpy.mean(pitch_lf_deg[idx_start:idx_end]))

        # SD of pitch during the sub-glide
        # TODO just use original radian array, not deg
        #      make sure "original" is radians ;)
        glide['SD_pitch'][i] = numpy.std(pitch_lf_deg[idx_start:idx_end]) * 180 / numpy.pi

        # mean temperature during the sub-glide
        glide['mean_temp'][i] = numpy.mean(temperature[idx_start:idx_end])

        # mean seawater density (kg/m^3) during the sub-glide
        glide['mean_swdensity'][i] = numpy.mean(Dsw[idx_start:idx_end])

        try:
            # TODO this will always go to except, need to test
            xpoly = numpy.arange(idx_start, idx_end)
            ypoly = swim_speed[xpoly]

            B, BINT, R, RINT, STATS = regress(ypoly, [xpoly,
                                                      numpy.ones(len(ypoly), 1)])

            # mean acceleration during the sub-glide
            glide['mean_acceleration'][i] = B[0]

            # R2-value for the regression swim speed vs. time during the sub-glide
            glide['R2_speed_vs_time'][i] = STATS[0]

            # SE of the gradient for the regression swim speed vs. time during the
            # sub-glide
            glide['SE_speed_vs_time'][i] = STATS[3]

        except:

            # mean acceleration during the sub-glide
            glide['mean_acceleration'][i] = numpy.nan

            # R2-value for the regression swim speed vs. time during the sub-glide
            glide['R2_speed_vs_time'][i] = numpy.nan

            # SE of the gradient for the regression swim speed vs. time during the
            # sub-glide
            glide['SE_speed_vs_time'][i] = numpy.nan


        # Dive phase:0 bottom, -1 descent, 1 ascent, NaN not dive phase
        sumphase = sum(phase[idx_start:idx_end])
        sp = numpy.nan
        if sumphase < 0:
            sp = -1
        elif sumphase == 0:
            sp = 0
        elif sumphase > 0:
            sp = 1
        glide['dive_phase'][i] = sp

        # TODO eliminate D, too many random tables
        D_ind = numpy.where((D[: , 0]*fs_a < idx_start) & (D[: , 1]*fs_a > idx_end))[0]

        if D_ind.size == 0:
            Dinf = numpy.zeros(D.shape[1])
            Dinf[:] = numpy.nan
        else:
            Dinf = D[D_ind, :][0]

        # Dive number in which the sub-glide recorded
        glide['dive_number'][i] = Dinf[6]

        # Maximum dive depth (m) of the dive
        glide['dive_max_depth'][i] = Dinf[5]

        # Dive duration (s) of the dive
        glide['dive_duration'][i] = Dinf[2]

        # Mean pitch(deg) calculated using circular statistics
        glide['mean_pitch_circ'][i] = astropy.stats.circmean(pitch_lf[idx_start:idx_end])

        # Measure of concentration (r) of pitch during the sub-glide (i.e. 0 for
        # random direction, 1 for unidirectional)
        glide['pitch_concentration'][i] = 1 - astropy.stats.circvar(pitch_lf[idx_start:idx_end])

        # Mean roll (deg) calculated using circular statistics
        glide['mean_roll_circ'][i] = astropy.stats.circmean(roll_lf[idx_start:idx_end])

        # Measure of concentration (r) of roll during the sub-glide
        glide['roll_concentration'][i] = 1 - astropy.stats.circvar(roll_lf[idx_start:idx_end])

        # Mean heading (deg) calculated using circular statistics
        glide['mean_heading_circ'][i] = astropy.stats.circmean(heading_lf[idx_start:idx_end])

        # Measure of concentration (r) of heading during the sub-glide
        glide['heading_concentration'][i] = 1 - astropy.stats.circvar(heading_lf[idx_start:idx_end])

    return glide


def calc_glide_ratios(dive_ind, des, asc, gl_mask, depths, pitch_lf):
    import numpy
    import pandas

    # TODO gl_ratio as numpy record array
    gl_ratio = numpy.zeros((dive_ind.shape[0], 10))

    keys = ['des_duration',
            'des_gl_duration',
            'des_gl_ratio',
            'des_mean_pitch',
            'des_rate',
            'asc_duration',
            'asc_gl_duration',
            'asc_gl_ratio',
            'asc_mean_pitch',
            'asc_rate',]

    gl_ratio = pandas.DataFrame(index=range(len(dive_ind)), columns=keys)

    # For each dive with start/stop indices in dive_ind
    for i in range(len(dive_ind)):
        # Get indices for descent and ascent phase of dive `i`
        des_ind = numpy.where(des[dive_ind[i,0]:dive_ind[i,1]])[0]
        asc_ind = numpy.where(asc[dive_ind[i,0]:dive_ind[i,1]])[0]

        # DESCENT
        # total duration of the descet phase (s)
        gl_ratio['des_duration'][i] = len(des_ind)

        # total glide duration during the descet phase (s)
        des_glides = numpy.where(gl_mask[des_ind])[0]
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
        asc_glides = numpy.where(gl_mask[asc_ind])[0]
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

