def EstimateDsw(SWdensity, depCTD, p):
    '''Estimate seawater density from CTD measurement

    Args
    ----
    SWdensity: seawater density from CTD measurement
    depCTD: CTD's depth data where SWdensity was recorded
    p: animal's depth data

    Returns
    -------
    Dsw: density of seawater
    '''
    return interp1(depCDT, SWdensity, p)


def SWdensityFromCTD(DPT, TMP, SL, D):
    '''Calculate seawater density at CTD depth'''
    import numpy

    import utils

    new_depth = range(max(DPT))

    temp = numpy.zeros(len(new_depth))
    sali = numpy.zeros(len(new_depth))
    temp[:] = numpy.nan
    sali[:] = numpy.nan

    temp[round(DPT)] = TMP
    sali[round(DPT)] = SL

    # linear interpret between the NaNs
    temp = utils.fixgaps(temp)
    sali = utils.fixgaps(sali)
    dens = sw_dens0(sali, temp)

    # TODO remove?
    plt.plot(dens, new_depth)
    ylabel('Depth (m)')
    xlabel('Density (kg/m^3)')

    if max(D[:, 6]) > max(DPT):
        new_depth2 = range(0, max(D[:, 6]))
        sali = numpy.hstack(sali, sali[-1] * numpy.ones((len(new_depth2)-len(new_depth))))
        temp = numpy.hstack(temp, temp[-1] * numpy.ones((len(new_depth2)-len(new_depth)))

        dens = sw_dens0(sali, temp)

        # TODO remove?
        plt.plot(dens, new_depth2)
        ylabel('Depth (m)')
        xlabel('Density (kg/m^3)')

    depth_CTD = numpy.copy(new_depth2)
    SWdensity = numpy.copy(dens)

    return SWdensity, depth_CTD


def sw_dens0(S, T):
    '''Density of Sea Water at atmospheric pressure

    Using UNESCO 1983 (Eqn 13, p.17) (EOS 1980) polynomial.

    Args
    ----
    S: salinity    [psu      (PSS-78)]
    T: temperature [degree C (ITS-90)]

    Returns
    -------
    dens0: density  [kg/m^3] of salt water with properties S,T,
    P:     0 (0 db gauge pressure)

    Authors
    -------
    Phil Morgan 92-11-05
    Lindsay Pender (Lindsay.Pender@csiro.au)

    References
    ----------
    Unesco 1983. Algorithms for computation of fundamental properties of
    seawater, 1983. _Unesco Tech. Pap. in Mar. Sci._, No. 44, 53 pp.

    Millero, F.J. and  Poisson, A.
    International one-atmosphere equation of state of seawater.
    Deep-Sea Res. 1981. Vol28A(6) pp625-629.
    '''

    mS, nS = S.shape
    mT, nT = T.shape

    if (mS != mT) | (nS != nT):
        raise SystemError('sw_dens0.py: S,T inputs must have the same '
                          'dimensions')
    T68 = T * 1.00024

    # UNESCO 1983

    # DEFINE CONSTANTS
    b0 =  0.824493
    b1 = -0.0040899
    b2 =  7.6438e-05
    b3 = -8.2467e-07
    b4 =  5.3875e-09
    c0 = -0.00572466
    c1 =  0.00010227
    c2 = -1.6546e-06
    d0 =  0.00048314

    dens = sw_smow(T) + \
           ((b0 + ((b1 + ((b2 + ((b3 + dot(b4, T68)) * T68)) * T68)), T68)) * S) + \
           (((c0 + multiply((c1 + (c2 * T68)) * T68)) * S) * numpy.sqrt(S)) + \
           (d0 * S**2)

    return dens


def sw_smow(T):
    '''Denisty of Standard Mean Ocean Water (Pure Water) using EOS 1980.

    Args
    ----
    T: temperature [degree C (ITS-90)]

    Returns
    -------
    dens = density  [kg/m^3]

    Authors
    -------
    Phil Morgan 92-11-05
    Lindsay Pender (Lindsay.Pender@csiro.au)

    References
    ----------
    Unesco 1983. Algorithms for computation of fundamental properties of
    seawater, 1983. _Unesco Tech. Pap. in Mar. Sci._, No. 44, 53 pp.
    UNESCO 1983 p17  Eqn(14)

    Millero, F.J & Poisson, A.
    INternational one-atmosphere equation of state for seawater.
    Deep-Sea Research Vol28A No.6. 1981 625-629.    Eqn (6)
    '''

    # DEFINE CONSTANTS
    a0 = 999.842594
    a1 =   0.06793952
    a2 =  -0.00909529
    a3 =   0.0001001685
    a4 =  -1.120083e-06
    a5 =   6.536332e-09

    T68 = T * 1.00024

    dens = a0 + \
           ((a1 + ((a2 + ((a3 + ((a4 + (a5 * T68)) * T68)), T68)) * T68)) * T68)

    return dens
