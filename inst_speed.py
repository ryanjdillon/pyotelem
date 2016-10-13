
def inst_speed(p, smoothpitch, fs, FR, f, k, thdeg, *args, **kwargs):
    nargin = sys._getframe(1).f_locals['nargin']
    varargin = sys._getframe(1).f_locals['varargin']
    nargout = sys._getframe(1).f_locals['nargout']

    #    [SwimSp] = inst_speed(p,smoothpitch,fs,FR,f,k,thdeg)
#    Estimate instantaneous swimming speed as depthrate
#
#    INPUT:
#       p = the depth time series in meters, sampled at fs Hz.
#       smoothpitch = pitch estimated from the low pass filtered
#       acceleration signal
#       fs = sensor sampling rate in Hz
#       FR = equal to the nominal stroking rate in Hz. Default value is 0.5
#           Hz. Use [] to get the default value if you want to enter
#           subsequent arguments.
#       f = number that multiplied by the FR gives the cut-off frequency fl,
#           of the low pass filter. f is a fraction of FR e.g., 0.4.
#       k = sample range over which to analyse.
#       thdeg = degrees threshold for which the speed will be estimated

    #    OUTPUT:
#          Instspeed = instantaneous speed calculated as the depthrate.

    #   Lucia Martina Martin Lopez (May 2016)
#   lmml2@st-andrews.ac.uk

    # define the cut off frequency (fc) for the low-pass filter
    fc = (dot(f, FR))
    # fl is the filter cut-off normalized to half the sampling frequency
# (Nyquist frequency).
    fl = fc / (fs / 2)
    # define the length of symmetric FIR (Finite Impulse Response) filter.
    nf = round(dot(fs / fc, 4))
    # apply a symmetric FIR low-pass filter to Aw and Mw with 0 group delay to
# obtain the low-pass filtered acceleration signal Alf

    newspeed = dot(diff(p[k]), fs)
    Instspeed = fir_nodelay(newspeed, nf, fl)
    InstSpeed = matlabarray(cat([Instspeed], [NaN]))
    SwimSp = - InstSpeed / smoothpitch
    xa = copy(smoothpitch)
    SwimSp[abs(dot(xa, 180) / pi) < thdeg] = NaN
