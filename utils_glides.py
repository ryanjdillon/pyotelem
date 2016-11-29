
def get_sgl_type(depths, fs, GL):
    '''Get phase of subglides, depth at glide, duration, times'''
    import numpy

    import utils
    t = numpy.arange(len(depths))/fs

    # Accelerometer (orginally in addition to magnetometer method)
    GLdur             = GL[:,1] - GL[:,0]
    GLT               = numpy.vstack([GL[:,0], GLdur]).T
    gl_ind, _         = utils.event_on(GLT, t)

    # sgl_type indicates whether it is stroking (1) or gliding(0)
    sgl_type = numpy.copy(gl_ind)
    sgl_type[gl_ind == 0] = 1
    sgl_type[gl_ind < 0]  = 0

    return sgl_type, gl_ind


def plot_get_fluke(depths, pitch, A_g_hf, T, nn, fs):
    import matplotlib.pyplot as plt

    import utils

    # Get max acc from 0:nn over axes [0,2] (x, z)
    maxy = numpy.max(numpy.max(Ahf[round(T[0, 0] * fs): round(T[nn, 1] * fs), [0,2]]))

    # Selection plot of depths an HPF x & z axes
    ax1 = plt.gca()
    ax2 = ax1.twiny()

    ax1.plot(range(len(depths)), depths, label='depths')
    ax2.plot(range(len(pitch)), Ahf[:, 0], lable='HPF x')
    ax2.plot(range(len(pitch)), Ahf[:, 2], lable='HPF z')

    ax1.invert_yaxis()

    for ax in [ax1, ax2]:
        ax.set_ylim(-2*maxy, 2*maxy)
        #ax.ytick(round(-2*maxy*10) / 10: 0.1: 2 * maxy)

    plt.legend(loc='upper right')
    plt.title('Zoom in to find thresholds for fluking, then enter it for J')

    plt.show()

    # Get user input for J
    J = utils.recursive_input('J (fluke magnitude)', float)

    return J


def plot_hf_acc_histo(Ahf, fs_a, stroke_f, DES, ASC):
    '''Get user selection of J from plot of histograms of high-pass filtered
    accelerometry

    Args
    ----
    Ahf: numpy.ndarray, shape(n,3)
        triaxial accelerometer data: index 1=x, 2=y, 3=z
    fs_a: int
        sampling rate (Hz)
    stroke_f: float
        stroke frequency (Hz)

    Returns
    -------
    J: float
        magnitude threshold for detecting a fluke stroke in m/s2.  If J is not
        given, fluke strokes will not be located but the rotations signal (pry)
        will be computed.
    '''
    import matplotlib.pyplot as plt
    import numpy

    import utils

    def plot_acc_distribution(ax, A_des, A_asc, title=''):
        # Group all descent and ascent data together for binning strokes
        TOTAL = numpy.abs(numpy.hstack([A_asc, A_des]).T)

        # TODO from .m code
        # Split into bins of 2*number of samples per stroke
        #Y, yb = utils.buffer(TOTAL, 2 * round(1/stroke_f*fs_a))
        #Ymax = numpy.sort(numpy.max(Y, axis=0))
        #ax.hist(numpy.max(Y, axis=0), 100)

        # Plot distribution of all acceleration samples
        Ymax = numpy.sort(TOTAL)
        ax.plot(range(len(Ymax)), Ymax)

        ax.set_ylim(0, numpy.max(Ymax))
        ax.title.set_text(title)

        return ax

    # Make histograms to review and select J from
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # HPF x acc. ascent and descent
    ax1 = plot_acc_distribution(ax1, Ahf[ASC, 0], Ahf[DES, 0], title='hpf-x acc')

    # HPF y acc. ascent and descent
    ax2 = plot_acc_distribution(ax2, Ahf[ASC, 2], Ahf[DES, 2], title='hpf-z acc')

    plt.show()

    # Get user selection for J
    J = utils.recursive_input('J (fluke magnitude)', float)

    return J


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


    def manual_freq(x, fs_a, nperseg, peak_thresh):
        '''Manually look at plot, then enter cutoff frequency for x,y,z'''
        import matplotlib.pyplot as plt

        import utils
        import utils_signal

        f_welch, S, _, _ = utils_signal.calc_PSD_welch(x, fs_a, nperseg)

        peak_loc, peak_mag = utils_signal.peakfinder(S, sel=None, peak_thresh=peak_thresh)
        peak_idx = peak_loc[peak_mag == max(peak_mag)]

        # Plot power specturm against frequency distribution
        plt.plot(f_welch, S)
        plt.scatter(f_welch[peak_loc], S[peak_loc], label='peaks')
        plt.legend(loc='upper right')
        plt.xlabel('Fequency (Hz)')
        plt.ylabel('"Power" (g**2 Hz**−1)')
        plt.show()

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
    # 1: dorsa-ventral
    # 2: lateral
    stroke_axes = [1, 2]

    # Lists for appending values from each axis
    cutoff_all   = list()
    stroke_f_all = list()
    f_all        = list()

    # Iterate over axes in `stroke_axes` list appending output to above lists
    for i in stroke_axes:
        # Auto calculate, else promter user for value after inspecting PSD plot
        if auto_select == True:
            cutoff, stroke_f = automatic_freq(x, fs_a, nperseg, peak_thresh)
        elif auto_select == False:
            cutoff, stroke_f =  manual_freq(x[:,i], fs_a, nperseg, peak_thresh)

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

    KK: 1-d ndarray
        matrix of cues to zero crossings in seconds (1st column) and
        zero-crossing directions (2nd column). +1 means a positive-going
        zero-crossing. Times are in seconds.

    Note
    ----
    If no J or t_max is given, J=[], or t_max=[], GL and KK returned as None

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

    # Find cues to each zero-crossing in vector pry(:,n), rotations around
    # the n axis.
    zc = utils_signal.findzc(A_g_hf_n, J, (t_max* fs_a) / 2)

    # find glides - any interval between zeros crossings greater than tmax
    ind = numpy.where(zc[1:, 0] - zc[0:-1, 1] > fs_a*t_max)[0]
    gl_ind = numpy.vstack([zc[ind, 0] - 1, zc[ind + 1, 1] + 1]).T

    # Compute mean index position of glide, and typecast to int for indexing
    # Shorten the glides to only include sections with jerk < J
    gl_c = numpy.round(numpy.mean(gl_ind, 1)).astype(int)
    gl_ind = numpy.round(gl_ind).astype(int)

    # TODO Remove if not necessary
    # Lambda function: return 0 if array has no elements, else returns first element
    #get_1st_or_zero = lambda x: x[0] if x.size != 0 else 0
            #over_J1 = get_1st_or_zero(over_J1)
            #over_J2 = get_1st_or_zero(over_J2)

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

    # convert sample numbers to times in seconds
    # TODO zc[:, 2] could not be sign by the 4th col in zero-crossing K
    KK = numpy.vstack((numpy.mean(zc[:, 0:1], 1) / fs_a, zc[:, 2])).T

    GL = gl_ind / fs_a
    GL = GL[numpy.where(GL[:, 1] - GL[:, 0] > t_max / 2)[0], :]

    return GL, KK



def split_glides(dur, GL, min_dur=False):
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

    # If minimum duration not passed, set to `min_dur` to skip slices < `dur`
    if min_dur == False:
        min_dur = dur

    # GL plus column for total duration of glide
    gl_ind_diff = numpy.vstack((GL.T, GL[:, 1] - GL[:, 0])).T

    # Split all glides in GL
    sgl_ind_started = False
    for i in range(len(GL)):
        gl_dur = gl_ind_diff[i, 2]

        # Split into sub glides if longer than duration
        if abs(gl_dur) > dur:

            # Make list of index lengths to start of each sub-glide
            n_sgl     = int(gl_dur//dur)
            sgl_dur   = numpy.ones(n_sgl)*dur
            sgl_start = numpy.arange(n_sgl)*(dur+1)

            # Add remainder as a sub-glide, skips if `min_dur` not passed
            if (gl_dur%dur > min_dur):
                last_dur = numpy.floor(gl_dur%dur)
                sgl_dur  = numpy.hstack([sgl_dur, last_dur])

                last_start = (len(sgl_start)*dur) + dur
                sgl_start  = numpy.hstack([sgl_start, last_start])

            # Get start and end index positions for each sub-glide
            for k in range(len(sgl_start)):

                # TODO round these to ints? or they are times...
                # starting at original glide start...
                # sgl_start_ind: add index increments of dur+1 for next start idx
                next_start_ind = gl_ind_diff[i, 0] + sgl_start[k]

                # end_glide: add `dur` to that to get ending idx
                next_end_ind = next_start_ind + sgl_dur[k]

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

    return sgl_ind


def calc_glide_des_asc( depths, fs, pitch_lf, roll_lf, heading_lf, swim_speed,
        D, phase, dur, sgl_ind, pitch_lf_deg, temperature, Dsw):
    '''Calculate glide ascent and descent summary data

    Args
    ----
    sgl_ind: numpy.ndarray, shape (n,2)
        start and end index positions for sub-glides
    '''
    import astropy.stats
    import numpy

    # TODO make numpy record array?
    glide = numpy.zeros((len(sgl_ind), 24))

    for i in range(len(sgl_ind)):
        idx_start = int(round(sgl_ind[i,0] * fs))
        idx_end = int(round(sgl_ind[i,1] * fs))

        # sub-glide start point in seconds
        glide[i,0] = sgl_ind[i,0]

        # sub-glide end point in seconds
        glide[i,1] = sgl_ind[i,1]

        # sub-glide duration
        glide[i,2] = sgl_ind[i,1] - sgl_ind[i,0]

        # mean depth(m)during sub-glide
        glide[i,3] = numpy.mean(depths[idx_start:idx_end])

        # total depth(m)change during sub-glide
        glide[i,4] = abs(depths[idx_start] - depths[idx_end])

        # mean swim speed during the sub-glide, only given if pitch>30 degrees
        glide[i,5] = numpy.mean(swim_speed[idx_start:idx_end])

        # mean pitch during the sub-glide
        glide[i,6] = numpy.mean(pitch_lf_deg[idx_start:idx_end])

        # mean sin pitch during the sub-glide
        glide[i,7] = numpy.sin(numpy.mean(pitch_lf_deg[idx_start:idx_end]))

        # SD of pitch during the sub-glide
        # TODO just use original radian array, not deg
        #      make sure "original" is radians ;)
        glide[i,8] = numpy.std(pitch_lf_deg[idx_start:idx_end]) * 180 / numpy.pi

        # mean temperature during the sub-glide
        glide[i,9] = numpy.mean(temperature[idx_start:idx_end])

        # mean seawater density (kg/m^3) during the sub-glide
        glide[i,10] = numpy.mean(Dsw[idx_start:idx_end])

        # TODO check matlab here
        try:
            xpoly = numpy.arange(idx_start, idx_end)
            ypoly = swim_speed[xpoly]

            B, BINT, R, RINT, STATS = regress(ypoly, [xpoly,
                                                      numpy.ones(len(ypoly), 1)])

            # mean acceleration during the sub-glide
            glide[i,11] = B[0]

            # R2-value for the regression swim speed vs. time during the sub-glide
            glide[i,12] = STATS[0]

            # SE of the gradient for the regression swim speed vs. time during the
            # sub-glide
            glide[i,13] = STATS[3]

        except:

            # mean acceleration during the sub-glide
            glide[i,11] = numpy.nan

            # R2-value for the regression swim speed vs. time during the sub-glide
            glide[i,12] = numpy.nan

            # SE of the gradient for the regression swim speed vs. time during the
            # sub-glide
            glide[i,13] = numpy.nan


        # Dive phase:0 bottom, -1 descent, 1 ascent, NaN not dive phase
        sumphase = sum(phase[idx_start:idx_end])
        sp = numpy.nan
        if sumphase < 0:
            sp = -1
        elif sumphase == 0:
            sp = 0
        elif sumphase > 0:
            sp = 1
        glide[i,14] = sp

        D_ind = numpy.where((D[: , 0]*fs < idx_start) & (D[: , 1]*fs > idx_end))[0]

        if D_ind.size == 0:
            Dinf = numpy.zeros(D.shape[1])
            Dinf[:] = numpy.nan
        else:
            Dinf = D[D_ind, :][0]

        # Dive number in which the sub-glide recorded
        glide[i,15] = Dinf[6]

        # Maximum dive depth (m) of the dive
        glide[i,16] = Dinf[5]

        # Dive duration (s) of the dive
        glide[i,17] = Dinf[2]

        # Mean pitch(deg) calculated using circular statistics
        glide[i,18] = astropy.stats.circmean(pitch_lf[idx_start:idx_end])

        # Measure of concentration (r) of pitch during the sub-glide (i.e. 0 for
        # random direction, 1 for unidirectional)
        glide[i,19] = 1 - astropy.stats.circvar(pitch_lf[idx_start:idx_end])

        # Mean roll (deg) calculated using circular statistics
        glide[i,20] = astropy.stats.circmean(roll_lf[idx_start:idx_end])

        # Measure of concentration (r) of roll during the sub-glide
        glide[i,21] = 1 - astropy.stats.circvar(roll_lf[idx_start:idx_end])

        # Mean heading (deg) calculated using circular statistics
        glide[i,22] = astropy.stats.circmean(heading_lf[idx_start:idx_end])

        # Measure of concentration (r) of heading during the sub-glide
        glide[i,23] = 1 - astropy.stats.circvar(heading_lf[idx_start:idx_end])

    return glide


def calc_glide_ratios(T, fs, bottom, SGtype, pitch_lf):
    import numpy

    # TODO gl_ratio as numpy record array
    gl_ratio = numpy.zeros((T.shape[0], 10))

    for dive in range(len(T)):
        # it is selecting the whole dive
        kk_des = numpy.arange(round(fs * T[dive, 0]),
                             round(fs * bottom[dive, 0]), dtype=int)

        # it is selecting the whole dive
        kk_as = numpy.arange(round(fs * bottom[dive,2] ),
                             round(fs * T[dive,1]), dtype = int)

        # total duration of the descet phase (s)
        gl_ratio[dive, 0] = len(kk_des) / fs

        # total glide duration during the descet phase (s)
        gl_ratio[dive, 1] = len(numpy.where(SGtype[kk_des] == 0)[0]) / fs

        # glide ratio during the descet phase
        gl_ratio[dive, 2] = gl_ratio[dive, 1] / gl_ratio[dive, 0]

        # mean pitch during the descet phase(degrees)
        gl_ratio[dive, 3] = numpy.mean(pitch_lf[kk_des] * 180 / numpy.pi)

        # descent rate (m/s)
        gl_ratio[dive, 4] = bottom[dive, 1] / gl_ratio[dive, 0]

        # total duration of the ascet phase (s)
        gl_ratio[dive, 5] = len(kk_as) / fs

        # total glide duration during the ascet phase (s)
        gl_ratio[dive, 6] = len(numpy.where(SGtype[kk_as] == 0)[0]) / fs

        # glide ratio during the ascet phase
        gl_ratio[dive, 7] = gl_ratio[dive, 6] / gl_ratio[dive, 5]

        # mean pitch during the ascet phase(degrees)
        gl_ratio[dive, 8] = numpy.mean(pitch_lf[kk_as] * 180 / numpy.pi)

        # ascent rate (m/s)
        gl_ratio[dive, 9] = bottom[dive, 2] / gl_ratio[dive, 5]

    return gl_ratio

