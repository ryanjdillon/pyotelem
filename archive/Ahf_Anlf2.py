
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


    Returns
    -------
    A_norm_lf: normalized low pass filtered 3-axis acceleration signal. It
               represents the slowly-varying postural changes, in m/s2.
               Normalization is to a field vector intensity of 1.  Ahf =
               high-pass filtered 3-axis acceleration signal, in m/s2.

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

    return A_norm_lf, A_norm_hf


