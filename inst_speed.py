
def inst_speed(depths, smoothpitch, fs, stroke_f, f, ind, thresh_deg):
    '''Estimate instantaneous swimming speed as depthrate

    Args
    ----
    depths:
        the depth time series in meters, sampled at fs Hz.
    smoothpitch:
        pitch estimated from the low pass filtered acceleration signal
    fs:
        sensor sampling rate in Hz
    stroke_f:
        equal to the nominal stroking rate in Hz. Default value is 0.5
        Hz. Use [] to get the default value if you want to enter
        subsequent arguments.
    f:
        number that multiplied by the stroke_f gives the cut-off frequency Wn,
        of the low pass filter. f is a fraction of stroke_f e.g., 0.4.
    ind:
        sample range over which to analyse.
    thresh_deg:
        degrees threshold for which the speed will be estimated

    Returns
    -------
    inst_speed:
        instantaneous speed calculated as the depth rate.

    Notes
    -----
    `p`  changed to `depths`
    `fl` changed to `Wn`
    `k`  changed to `ind`
    `Instspeed` changed to `inst_speed`
    `SwimSp` changed to `swim_speed`

    Lucia Martina Martin Lopez (May 2016)
    lmml2@st-andrews.ac.uk
    '''
    import numpy

    import utils_signal

    # TODO default stroke_f rate 0.5 here, 0.4 elsewhere

    # define the cut off frequency (cutoff) for the low-pass filter
    cutoff = f * stroke_f

    # Wn is the filter cut-off normalized to half the sampling frequency
    # (a.k.a the critical frequency).
    nyq = 0.5 * fs
    Wn  = cutoff / nyq

    # define the length of symmetric FIR (Finite Impulse Response) filter.
    n_f = round((fs / cutoff) * 4)

    # apply a symmetric FIR low-pass filter to Aw and Mw with 0 group delay to
    # obtain the low-pass filtered acceleration signal Alf
    new_speed = numpy.diff(depths[ind]) * fs

    inst_speed = utils_signal.fir_nodelay(new_speed, n_f, Wn)
    InstSpeed  = numpy.hstack((inst_speed, numpy.nan))

    swim_speed = -InstSpeed / smoothpitch
    xa         = numpy.copy(smoothpitch)
    swim_speed[abs(numpy.rad2deg(xa)) < thresh_deg] = numpy.nan

    # TODO
    # Docstring states inst_speed & swim_speed returned
    # [swim_speed] = inst_speed(depths, smoothpitch, fs, stroke_f, f, ind, thresh_deg)

    return swim_speed
