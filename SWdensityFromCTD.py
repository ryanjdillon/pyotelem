
def SWdensityFromCTD(DPT, TMP, SL, D, *args, **kwargs):
    nargin = sys._getframe(1).f_locals['nargin']
    varargin = sys._getframe(1).f_locals['varargin']
    nargout = sys._getframe(1).f_locals['nargout']

    newdepth = range(0, max(DPT))
    temp = NaN(size(newdepth))
    sali = NaN(size(newdepth))
    temp[round(DPT)] = TMP
    sali[round(DPT)] = SL
    # linear interpret between the NaNs
    temp = fixgaps(temp)
    sali = fixgaps(sali)
    dens = sw_dens0(sali, temp)
    figure(5)
    clf
    plot(dens, newdepth)
    ylabel('Depth (m)')
    xlabel('Density (kg/m^3)')
    if max(D[:, 6]) > max(DPT):
        newdepth2 = range(0, max(D[:, 6]))
        sali = matlabarray(
            cat(sali, (dot(sali[end()], ones((length(newdepth2) - length(newdepth)), 1))).T))
        temp = matlabarray(
            cat(temp, (dot(temp[end()], ones((length(newdepth2) - length(newdepth)), 1))).T))
        dens = sw_dens0(sali, temp)
        figure(5)
        clf
        plot(dens, newdepth2)
        ylabel('Depth (m)')
        xlabel('Density (kg/m^3)')

    depCTD = copy(newdepth2)
    SWdensity = copy(dens)
    return SWdensity, depCTD

if __name__ == '__main__':
    pass
