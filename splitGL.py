

def splitGL(dur, GL, *args, **kwargs):
    nargin = sys._getframe(1).f_locals['nargin']
    varargin = sys._getframe(1).f_locals['varargin']
    nargout = sys._getframe(1).f_locals['nargout']

    #   SGL=splitGL(dur,GL);
#   Make x duration subglides.

    #    INPUT:
#       dur = desired duration of glides
#       GL = matrix containing the start time (first column) and end time
#           (2nd column) of any glides.Times are in seconds.

    #    OUTPUT:
#          SGL = matrix containing the start time (first column) and end
#           time(2nd column) of the generated subglides.All glides must
#           have duration equal to the given dur value.Times are in seconds.

    #   Lucia Martina Martin Lopez (May 2016)
#   lmml2@st-andrews.ac.uk

    SUM = matlabarray([])
    for i in range(1, length(GL)).reshape(-1):
        sum = matlabarray([])
        GLINF = matlabarray(cat(GL, GL[:, 2] - GL[:, 1]))
        ng = (GLINF[i, 3] / (dur + 1))
        if abs(GLINF[i, 3]) > dur:
            STARTGL = matlabarray([])
            ENDGL = matlabarray([])
            for k in range(1, round(ng)).reshape(-1):
                v = dot(((range(1, max(round(ng)))) - 1), 6)
                startglide1 = GLINF[i, 1] + v[k]
                endglide1 = startglide1 + 5
                STARTGL = matlabarray(cat([STARTGL], [startglide1]))
                ENDGL = matlabarray(cat([ENDGL], [endglide1]))
            sum = matlabarray(cat(STARTGL, ENDGL))
        SUM = matlabarray(cat([SUM], [sum]))
        SGL = matlabarray(cat(SUM))
