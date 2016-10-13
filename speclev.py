
def speclev(x, nfft, fs, w, nov, *args, **kwargs):
    nargin = sys._getframe(1).f_locals['nargin']
    varargin = sys._getframe(1).f_locals['varargin']
    nargout = sys._getframe(1).f_locals['nargout']

    #    [SL,f]=speclev(x,nfft,fs,w,nov) S is the amount of power in each particular frequency (f)
#  x is the signal from which the speclev (power spectra) is going to be
#  calculated
# fft  is the fourier transform length
# fs is the frequency sample
# w is that if there is less than 4 inputs then w is equal to the nfft
# nov is that if there is less than 5 inputs then nov which is the overlap will be half of the size of the nfft
#    mark johnson, WHOI
#    mjohnson@whoi.edu
    if nargin < 2:
        nfft = 512

    if nargin < 3:
        fs = 1

    if nargin < logical_or(4, isempty(w)):
        w = copy(nfft)

    if nargin < 5:
        nov = nfft / 2

    if length(w) == 1:
        w = hanning(w)

    P = zeros(nfft / 2, size(x, 2))
    for k in range(1, size(x, 2)).reshape(-1):
        X, z = buffer(x[:, k], length(w), nov, 'nodelay', nargout=2)
        X = multiply(detrend(X), repmat(w, 1, size(X, 2)))
        F = abs(fft(X, nfft)) ** 2
        P[:, k] = sum(F[1:nfft / 2, :], 2)

    ndt = size(X, 2)
    # these two lines give correct output for randn input
# SL of randn should be -10*log10(fs/2)

    slc = 3 - dot(10, log10(fs / nfft)) - dot(10, log10(sum(w ** 2) / nfft))
    SL = dot(10, log10(P)) - dot(10, log10(ndt)) - dot(20, log10(nfft)) + slc
    f = dot((range(0, nfft / 2 - 1)) / nfft, fs)
