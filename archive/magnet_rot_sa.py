
def magnet_rot_sa(Aw, Mw, fs, FR, f, alpha, n, k, J=None, tmax=None):
    '''Estimate body rotations (pry) and specific acceleration (Sa) using 3-axis
    magnetometer and accelerometer sensors. Glides (GL) and strokes (KK) are
    identified.

    Args
    ----
    Aw:
        whale frame triaxial accelerometer matrix at sampling rate fs.
    Mw:
        whale frame triaxial magnetometer matrix at sampling rate fs.
    fs:
        sensor sampling rate in Hz
    FR:
        equal to the nominal stroking rate in Hz. Default value is 0.5 Hz.
        Use [] to get the default value if you want to enter subsequent
        arguments.
    f:
        number that multiplied by the FR gives the cut-off frequency fl,
        of the low pass filter. f is a fraction of FR e.g., 0.4.
    alpha:
        when the animal's rotation axis is within a few degrees of the Earth's
        magnetic field vector stroking rotations will have a poor signal to
        noise ratio and should be removed from analysis. alpha sets the angular
        threshold in degrees below which data will be removed. Default value is
        25 degrees.
    n:
        fundamental axis around which body rotations are analysed.
        1 for rotations around the y axis, pitch method.
        2 for rotations around the x axis, roll method.
        3 for rotations around the z axis, yaw method.
    k:
        sample range over which to analyse.
    J:
        magnitude threshold for detecting a fluke stroke in radians.
        If J is not given, fluke strokes will not be located but the rotations
        signal (pry) will be computed.If no J is given or J=[], no GL and KK
        output will be generated.
    tmax:
        maximum duration allowable for a fluke stroke in seconds.
        A fluke stroke is counted whenever there is a cyclic variation in the
        pitch deviation with peak-to-peak magnitude greater than +/-J and
        consistent with a fluke stroke duration of less than tmax seconds,
        e.g., for Mesoplodon choose tmax=4.  If no tmax is given or tmax=[], no
        GL and KK output will be generated.

    Returns
    -------
    MagAcc:
        structure array containing the following elements
        [Anlf,Mnlf,Ahf,Mhf,Mhf_est,Qt]
        Anlf:
            normalized low pass filtered 3-axis acceleration signal.  It
            represents the slowly-varying postural changes, in m/s2.
            Normalization is to a field vector intensity of 1.
        Mnlf:
            normalized low pass filtered 3 axis magnetometer signal. It
            represents the slowly-varying postural changes.  Normalization is
            to a field vector intensity of 1.
        Ahf:
            high-pass filtered 3-axis acceleration signal, in m/s2.
        Mhf:
            high-pass filtered 3-axis magnetometer signal.
        Mhf_est:
            estimate high-pass filtered magnetometer signal
        Qt:
            estimated orientation component of the dynamic acceleration.
    pry:
        estimated body rotations around the fundamental axis of rotation
        [pitch, roll, yaw] which depending on the locomotion style it can be
        around the y, x or z axis. Angles not estimated are set to 0. Angles
        are in radians.
    Sa:
        estimated specific acceleration [sx, sy, sz]in m/s2.
    GL:
        matrix containing the start time (first column) and end time (2nd
        column) of any glides (i.e., no zero crossings in tmax or more
        seconds).Times are in seconds.
    KK:
        matrix of cues to zero crossings in seconds (1st column) and
        zero-crossing directions (2nd column). +1 means a positive-going
        zero-crossing. Times are in seconds.

    NOTE
    ----
    Be aware that when using devices that combine different sensors as in here
    (accelerometer and magnetometer) their coordinate systems should be
    aligned. If they are not physically aligned, invert as necessary for all
    the senso's axes to be aligned so that when a positive rotation in roll
    occurs in one sensor it is also positive in all the sensors.

    Lucia Martina Martin Lopez & Mark Johnson  (June 2013)
    '''
    import numpy

    # define the cut off frequency (fc) for the low-pass filter
    fc = f * FR

    # fl is the filter cut-off normalized to half the sampling frequency
    # (Nyquist frequency).
    fl = fc / (fs / 2)

    # define the length of symmetric FIR (Finite Impulse Response) filter.
    nf = round(fs / fc * 4)

    # apply a symmetric FIR low-pass filter to Aw and Mw with 0 group delay to
    # obtain the low-pass filtered acceleration signal Alf and the low-pass
    # filtered magnetometer signal Mlf
    Alf = fir_nodelay(Aw, nf, fl)
    Mlf = fir_nodelay(Mw, nf, fl)

    # normalize the Mlf and Alf, the low-pass filtered Mw and Aw signals
    # respectively. By assumption, these should have a constant magnitude equal
    # to the magnetic field intensity and the gravity field intensity
    # respectively.
    NM = norm2(Mlf[k, :]) ** (- 1)
    NA = norm2(Alf[k, :]) ** (- 1)

    Mnlf = Mlf[k, :] * repmat(NM, 1, 3))
    Anlf = Alf[k, :] * repmat(NA, 1, 3))

    # normalize the magnetometer - field intensity is not needed.  normalize
    # Aw, to be consistent with the following subtraction equation and
    # calculate the normalized high-pass filtered Aw signal.
    Mnw = multiply(Mw[k, :], repmat(NM, 1, 3))
    Anw = multiply(Aw[k, :], repmat(NA, 1, 3))

    # calculate Mhf and Ahf, the normalized high-pass filtered Mw and Aw
    # signals respectively.
    Mhf = Mnw - Mnlf
    Ahf = Anw - Anlf

    x = Mnlf[:,0]
    y = Mnlf[:,1]
    z = Mnlf[:,2]

    if n == 1:
        print('you chose for rotations around the y axis, pitch method ')

        # pitching rotation. rotation around the y axis.  find least
        # squared-error estimator to estimate body rotations
        c = x**2 + z**2
        ic = 1.0 / c

        W = numpy.hstack((z*ic, numpy.zeros(len(k), 1), -x*ic))

        mp = (Mhf * W) * numpy.ones(3)

        pry = numpy.hstack((float(numpy.arcsin(mp)), numpy.zeros(len(k), 2) ))

        # remove data points when the animals lateral axis (y) is within alpha
        # degrees of the Earth's magnetic field vector.
        kk = numpy.where(x**2 + z**2 < \
                         y**2 * ((1 / numpy.cos(alpha * numpy.pi/180) ** 2) - 1))[0]

        pry[kk, :] = numpy.nan
    elif n == 2:
        print('you chose for rotations around the x axis, roll method')

        # rolling rotation. rotation around the x axis.  find least
        # squared-error estimator to estimate body rotations
        c = y**2 + z**2
        ic = 1.0 / c
        W = numpy.hstack((numpy.zeros(len(k), 1), ( z * ic), (-y * ic)))

        mp = (Mhf* W) * numpy.ones(3)

        pry = numpy.hstack((numpy.zeros(len(k), 1),
                            float(numpy.arcsin(mp)),
                            numpy.zeros(len(k), 1)
                            ))

        # remove data points when the animals longitudinal axis (x)is
        # within alpha degrees of the Earth's magnetic field vector.
        kk = numpy.where( y**2 + z**2 < \
                          x**2 * ((1/numpy.cos(alpha * numpy.pi/180) ** 2) - 1))[0]

        pry[kk, :] = numpy.nan

    elif n == 3:
        print('you chose for rotations around the z axis, yaw method')

        # yaw rotation. rotation around the z axis.
        # find least squared-error estimator to estimate body rotations
        # this corresponds to Eqn 7 of (Martin et al., 2015).
        c = x**2 + y**2
        ic = 1.0 / c
        W = numpy.hstack((y*ic, -x*ic, numpy.zeros(len(k), 1) ))

        mp = (Mhf * W) * numpy.ones(3)
        pry = numpy.hstack((numpy.zeros(len(k), 2), float(numpy.arcsin(mp)) ))

        # remove data points when the animals dorso-ventral axis (z) is
        # within alpha degrees of the Earth's magnetic field vector.
        kk = numpy.where( x**2 + y**2 < \
                          z**2 * ((1 / numpy.cos(alpha * numpy.pi/180) ** 2) - 1))[0]

        pry[kk, :] = numpy.nan

    sinpry = numpy.sin(pry)

    # estimate high-pass filtered magnetometer signal assuming that body
    # rotations occur mainly around one body axis. E.g., body rotations during
    # stroking around the y axis will generate signals in both the x and the z
    # axis of the magnetometer.
    if n == 1:
        Mhf_est = numpy.hstack((z * sinpry[:, n],
                                numpy.zeros(Mnw.shape[0], 1),
                               -x * sinpry[:, n]))
    elif n == 2:
        Mhf_est = numpy.hstack((numpy.zeros(Mnw.shape[0], 1),
                                z * sinpry[:, n],
                               -y * sinpry[:, n]))
    elif n == 3:
        Mhf_est = numpy.hstack((y * sinpry[ :, n],
                               -x * sinpry[:, n],
                                numpy.zeros(Mnw.shape[0], 1)))


    # estimate the orientation component of the dynamic acceleration Oda
    if n == 1:
        Oda = numpy.hstack((Anlf[:, 3] * sinpry[:, n],
                            numpy.zeros(Mnw.shape[0], 1),
                           -Anlf[:, 1] * sinpry[:, n]))
    elif n == 2:
        Oda = (cat(numpy.zeros(size(Mnw, 1), 1), multiply(
            Anlf[:, 3], sinpry[:, n]), multiply(- Anlf[:, 2], sinpry[:, n])))
    elif n == 3:
        Oda = (cat(multiply(Anlf[:, 2], sinpry[ :, n]), multiply(-
            Anlf[:, 1], sinpry[:, n]), numpy.zeros(size(Mnw, 1), 1)))

    Sa = 9.81 * (Ahf - Oda)

    field1 = 'Anlf'
    value1 = Anlf*9.81
    field2 = 'Mnlf'
    value2 = copy(Mnlf)
    field3 = 'Ahf'
    value3 = Ahf*9.81
    field4 = 'Mhf'
    value4 = copy(Mhf)
    field5 = 'Mhf_est'
    value5 = copy(Mhf_est)
    field6 = 'Qt'
    value6 = Oda*9.81

    MagAcc = struct( field1, value1, field2, value2, field3, value3, field4,
            value4, field5, value5, field6, value6)




    if isempty(J) | isempty(tmax):
        GL = None
        KK = None
        print( 'Cues for strokes(KK) and glides (GL) are not given as J and '
               'tmax are not set')
    else:
        # Find cues to each zero-crossing in vector pry(:,n), rotations around the
        # n axis.
        K = findzc(pry[:, n], J, tmax*fs / 2)

        k = find(K[2:end(), 1] - K[1:end() - 1, 2] > fs*tmax)
        glk = matlabarray(cat(K[k, 1] - 1, K[k + 1, 2] + 1))

        glc = round(mean(glk, 2))
        for k in range(1, len(glc)).reshape(-1):
            kk = range(glc[k], glk[k, 1], - 1)
            test = find(isnan(pry[kk, n]))
            if logical_not(isempty(test)):
                glc[k] = numpy.nan
                glk[k, 1] = numpy.nan
                glk[k, 2] = numpy.nan
            else:
                glk[k, 1] = glc[k] - find(abs(pry[kk, n]) >= J, 1) + 1
                kk = range(glc[k], glk[k, 2])
                glk[k, 2] = glc[k] + find(abs(pry[kk, n]) >= J, 1) - 1

        # convert sample numbers to times in seconds
        KK = matlabarray(cat(mean(K[:, 1:2], 2) / fs, K[:, 3]))
        GL = glk / fs

        GL = GL[find(GL[:, 2] - GL[:, 1] > tmax / 2), :]

    return MagAcc, pry, Sa, GL, KK
