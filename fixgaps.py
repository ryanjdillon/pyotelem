

def fixgaps(x, *args, **kwargs):
    nargin = sys._getframe(1).f_locals['nargin']
    varargin = sys._getframe(1).f_locals['varargin']
    nargout = sys._getframe(1).f_locals['nargout']

    # FIXGAPS Linearly interpolates gaps in a time series
# YOUT=FIXGAPS(YIN) linearly interpolates over NaN
# in the input time series (may be complex), but ignores
# trailing and leading NaN.

    # R. Pawlowicz 6/Nov/99

    y = copy(x)
    bd = isnan(x)
    gd = find(logical_not(bd))
    bd[cat(range(1, (min(gd) - 1)), range((max(gd) + 1), end()))] = 0
    y[bd] = interp1(gd, x[gd], find(bd))
