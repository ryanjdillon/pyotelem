
def Ahf_Anlf(Aw, fs, FR, f, n, k, J, tmax, *args, **kwargs):
    nargin = sys._getframe(1).f_locals['nargin']
    varargin = sys._getframe(1).f_locals['varargin']
    nargout = sys._getframe(1).f_locals['nargout']

    #    [Anlf,Ahf,GL,KK] = Ahf_Anf(Aw,fs,FR,f,n,k,[],[])
#    Estimate low and high pass-filtered acceleration signals using 3-axis
#    accelerometer sensors.
#    [Anlf,Ahf,GL,KK] = Ahf_Anf(Aw,fs,FR,f,n,k,J,tmax)
#    Estimate low and high pass-filtered acceleration signals using 3-axis
#    accelerometer sensors. Glides (GL) and strokes(KK) are identified.
#
#    INPUT:
#       Aw = whale frame triaxial accelerometer matrix at sampling rate fs.
#       fs = sensor sampling rate in Hz
#       FR = equal to the nominal stroking rate in Hz. Default value is 0.5
#           Hz. Use [] to get the default value if you want to enter
#           subsequent arguments.
#       f = number that multiplied by the FR gives the cut-off frequency fl,
#           of the low pass filter. f is a fraction of FR e.g., 0.4.
#       n = fundamental axis of the acceleration signal.
#           1 for accelerations along the x axis, longitudinal axis.
#           2 for accelerations along the y axis, lateral axis.
#           3 for accelerations along the z axis, dorso-ventral axis.
#       k = sample range over which to analyse.
#       J = magnitude threshold for detecting a fluke stroke in m/s2.
#           If J is not given, fluke strokes will not be located
#           but the rotations signal (pry) will be computed.If no J is
#           given or J=[], no GL and KK output will be generated.
#       tmax = maximum duration allowable for a fluke stroke in seconds.
#           A fluke stroke is counted whenever there is a cyclic variation
#           in the pitch deviation with peak-to-peak magnitude
#           greater than +/-J and consistent with a fluke stroke duration
#           of less than tmax seconds, e.g., for Mesoplodon choose tmax=4.
#           If no tmax is given or tmax=[], no GL and KK output will be
#           generated.

    #    OUTPUT:
#           Anlf = normalized low pass filtered 3-axis acceleration signal.
#               It represents the slowly-varying postural changes, in m/s2.
#               Normalization is to a field vector intensity of 1.
#           Ahf = high-pass filtered 3-axis acceleration signal, in m/s2.
#       GL = matrix containing the start time (first column) and end time
#           (2nd column) of any glides (i.e., no zero crossings in tmax or
#           more seconds).Times are in seconds.
#       KK = matrix of cues to zero crossings in seconds (1st column) and
#           zero-crossing directions (2nd column). +1 means a
#           positive-going zero-crossing. Times are in seconds.

    #    NOTE: Be aware that when using devices that combine different sensors
#       as in here (accelerometer and magnetometer) their coordinate
#       systems should be aligned. If they are not physically aligned,
#       invert as necessary for all the senso's axes to be aligned so that
#       when a positive rotation in roll occurs in one sensor it is also
# `     positive in all the sensors.

    #   Lucia Martina Martin Lopez & Mark Johnson

    # define the cut off frequency (fc) for the low-pass filter
    fc = (dot(f, FR))
    # fl is the filter cut-off normalized to half the sampling frequency
# (Nyquist frequency).
    fl = fc / (fs / 2)
    # define the length of symmetric FIR (Finite Impulse Response) filter.
    nf = round(dot(fs / fc, 4))
    # apply a symmetric FIR low-pass filter to Aw with 0 group delay to
# obtain the low-pass filtered acceleration signal Alf
    Alf = fir_nodelay(Aw, nf, fl)
    # normalize the Alf, the low-pass filtered Aw signal. By assumption,
# it should have a constant magnitude equal to the gravity field intensity
    NA = norm2(Alf[k, :]) ** (- 1)
    Anlf = multiply(Alf[k, :], repmat(NA, 1, 3))
    # normalize Aw, to be consistent with the following subtraction equation and
# calculate the normalized high-pass filtered Aw signal.
    Anw = multiply(Aw[k, :], repmat(NA, 1, 3))
    # calculateAhf, the normalized high-pass filtered Aw signal.
    Ahf = Anw - Anlf
    Ahf = dot(Ahf, 9.81)
    Anlf = dot(Anlf, 9.81)
    if logical_or(isempty(J), isempty(tmax)):
        GL = matlabarray([])
        KK = matlabarray([])
        print(
            'Cues for strokes(KK) and glides (GL) are not given as J and tmax are not set')
        return Anlf, Ahf, GL, KK

    # Find cues to each zero-crossing in vector pry(:,n), rotations around the
 # n axis.
    K = findzc(Ahf[:, n], J, dot(tmax, fs) / 2)

    k = find(K[2:end(), 1] - K[1:end() - 1, 2] > dot(fs, tmax))
    glk = matlabarray(cat(K[k, 1] - 1, K[k + 1, 2] + 1))

    glc = round(mean(glk, 2))
    for k in range(1, length(glc)).reshape(-1):
        kk = range(glc[k], glk[k, 1], - 1)
        test = find(isnan(Ahf[kk, n]))
        if logical_not(isempty(test)):
            glc[k] = NaN
            glk[k, 1] = NaN
            glk[k, 2] = NaN
        else:
            glk[k, 1] = glc[k] - find(abs(Ahf[kk, n]) >= J, 1) + 1
            kk = range(glc[k], glk[k, 2])
            glk[k, 2] = glc[k] + find(abs(Ahf[kk, n]) >= J, 1) - 1

    # convert sample numbers to times in seconds
    KK = matlabarray(cat(mean(K[:, 1:2], 2) / fs, K[:, 3]))
    GL = glk / fs
    GL = GL[find(GL[:, 2] - GL[:, 1] > tmax / 2), :]
