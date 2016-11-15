
def Ahf_Anlf(A_afr, fs, stroke_f_hz, f_frac, n, col, J=None, t_max=None):
    '''
    Estimate low and high pass-filtered acceleration signals using 3-axis
    accelerometer sensors.

    Glides (GL) and strokes(KK) are identified if `J` and `t_max` are passed.

    Args
    ----
    A_afr:       whale frame triaxial accelerometer matrix at sampling rate fs.

    fs:          sensor sampling rate in Hz

    stroke_f_hz: equal to the nominal stroking rate in Hz. Default value is 0.5 Hz.
                 Use [] to get the default value if you want to enter
                 subsequent arguments.

    f_frac:      number that multiplied by the stroke_f_hz gives the cut-off
                 frequency fl, of the low pass filter. f_frac is a fraction of
                 stroke_f_hz e.g., 0.4.

    n:           fundamental axis of the acceleration signal.
                 1 for accelerations along the x axis, longitudinal axis.
                 2 for accelerations along the y axis, lateral axis.
                 3 for accelerations along the z axis, dorso-ventral axis.

    col:           sample range over which to analyse.

    J:           magnitude threshold for detecting a fluke stroke in m/s2.  If
                 J is not given, fluke strokes will not be located but the
                 rotations signal (pry) will be computed.If no J is given or
                 J=[], no GL and KK output will be generated.

    t_max:       maximum duration allowable for a fluke stroke in seconds.  A
                 fluke stroke is counted whenever there is a cyclic variation
                 in the pitch deviation with peak-to-peak magnitude greater
                 than +/-J and consistent with a fluke stroke duration of less
                 than t_max seconds, e.g., for Mesoplodon choose t_max=4.  If
                 no t_max is given or t_max=[], no GL and KK output will be
                 generated.

    Returns
    -------
    A_norm_lf: normalized low pass filtered 3-axis acceleration signal. It
               represents the slowly-varying postural changes, in m/s2.
               Normalization is to a field vector intensity of 1.  Ahf =
               high-pass filtered 3-axis acceleration signal, in m/s2.

    GL:        matrix containing the start time (first column) and end time
               (2nd column) of any glides (i.e., no zero crossings in t_max or
               more seconds).Times are in seconds.

    KK:        matrix of cues to zero crossings in seconds (1st column) and
               zero-crossing directions (2nd column). +1 means a positive-going
               zero-crossing. Times are in seconds.

    NOTE
    ----
    Be aware that when using devices that combine different sensors as in here
    (accelerometer and magnetometer) their coordinate systems should be
    aligned. If they are not physically aligned, invert as necessary for all
    the senso's axes to be aligned so that when a positive rotation in roll
    occurs in one sensor it is also positive in all the sensors.

    Lucia Martina Martin Lopez & Mark Johnson
    '''
    import utils
    import numpy
    import scipy.signal

    # define the cut off frequency (fc) for the low-pass filter
    cutoff_low = f_frac * stroke_f_hz

    # fl is the filter cut-off normalized to half the sampling frequency
    # (Nyquist frequency).
    nyq = fs / 2
    fl = cutoff_low / nyq

    # define the length of symmetric FIR (Finite Impulse Response) filter.
    nf = round((fs/cutoff_low) * 4)

    # apply a symmetric FIR low-pass filter to Aw with 0 group delay to
    # obtain the low-pass filtered acceleration signal A_lf
    b = scipy.signal.firwin(n, fl)
    A_lf = scipy.signal.lfilter(b, numpy_ones(len(b)), x)

    # normalize the A_lf, the low-pass filtered Aw signal. By assumption,
    # it should have a constant magnitude equal to the gravity field intensity
    # TODO sort normalization out
    utils.normalized(a, 0, order=2)
    NA = numpy.linalg.norm(A_lf[col, :]) ** (- 1)
    A_norm_lf = A_lf[col, :] * NA

    # normalize Aw, to be consistent with the following subtraction equation and
    # calculate the normalized high-pass filtered Aw signal.
    A_norm_afr = A_afr[col, :] * NA

    # calculate Ahf, the normalized high-pass filtered Aw signal.
    # NOTE: multiplication by gravity constant removed, apply this externally
    # if needed
    A_norm_hf = A_norm_afr - A_norm_lf

    if isempty(J) or isempty(t_max):
        GL = None
        KK = None
        print( 'Cues for strokes(KK) and glides (GL) are not given as J and '
               't_max are not set')
    else:
        # Find cues to each zero-crossing in vector pry(:,n), rotations around
        # the n axis.
        K = utils.findzc(A_norm_hf[:, n], J, (t_max* fs) / 2)

        k = numpy.where(K[1:-1, 0] - K[0:-2, 1] > fs*t_max)
        glk = numpy.hstack(K[k, 0] - 1, K[k + 1, 1] + 1)

        glc = round(numpy.mean(glk, 1))
        for i in range(len(glc)):
            col = range(glc[i], glk[i, 0], - 1)
            test = numpy.where(numpy.isnan(A_norm_hf[col, n]))
            if test.size != 0:
                glc[i]    = numpy.nan
                glk[i,0] = numpy.nan
                glk[i,1] = numpy.nan
            else:
                glk[i,0] = glc[i] - numpy.where(abs(A_norm_hf[col, n]) >= J) + 1
                col      = range(glc[i], glk[i, 1])
                glk[i,1] = glc[i] + numpy.where(abs(A_norm_hf[col, n]) >= J) - 1

        # convert sample numbers to times in seconds
        KK = numpy.hstack(numpy.mean(K[:, 0:1], 1) / fs, K[:, 2])
        GL = glk / fs
        GL = GL[numpy.where(GL[:, 1] - GL[:, 0] > t_max / 2), :]

    return A_norm_lf, A_norm_hf, GL, KK
