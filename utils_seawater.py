
def get_SST(datetimes, lons, lats):
    #import planetos

    #SST = retrieve_SST_from_somewhere(datetimes, lons.min(), lons.max(),
    #                                             lats.min(), lats.max())
    SST = None

    return SST

def estimate_seawater_density(depths, depth_ctd, temp_ctd, sali_ctd):
    '''Estimate seawater density

    From body condition script
    '''
    import utils_sewater

    tsd = SWdensityFromCTD(depth_ctd, temp_ctd, sali_ctd)

    # TODO handle case where depth greater than CTD max, return last value/NaN
    densities = tsd['density'][depths.round().astype(int)]

    return tsd, densities


def SWdensityFromCTD(depth_ctd, temp_ctd, sali_ctd, duplicates='last'):
    '''Calculate seawater density at CTD depth'''
    import numpy
    import pandas

    import utils

    # Create empty data frame incremented by whole meters to max CTD depth
    columns = ['temperature', 'salinity', 'density']
    n_samples = numpy.ceil(depth_ctd.max()).astype(int)
    tsd = pandas.DataFrame(index=range(n_samples), columns=columns)

    # Assign temperature and salinity for each rounded ctd depth
    depths = depth_ctd.round().astype(int)
    for d in numpy.unique(depths):
        # Use last or first occurance of depth/value pairs
        if duplicates == 'last':
            idx = numpy.where(depths == d)[0][-1]
        elif duplicates == 'first':
            idx = numpy.where(depths == d)[0][0]

        # Fill temperature and salinity at measured depths, rounded to whole meter
        tsd['temperature'][d] = temp_ctd[idx]
        tsd['salinity'][d]    = sali_ctd[idx]

    # Linearly interpolate temperature and salinity measurements
    tsd = tsd.astype(float)
    tsd.interpolate('linear', inplace=True)

    # Cacluate seawater density
    tsd['density'] = sw_dens0(tsd['salinity'][:], tsd['temperature'][:])

    return tsd


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
    import numpy

    if S.shape != T.shape:
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
           ((b0 + ((b1 + ((b2 + ((b3 + (b4 * T68)) * T68)) * T68)) * T68)) * S) + \
           (((c0 + ((c1 + (c2 * T68)) * T68)) * S) * numpy.sqrt(S)) + \
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
    a1 =   6.793952e-2
    a2 =  -9.095290e-3
    a3 =   1.001685e-4
    a4 =  -1.120083e-6
    a5 =   6.536332e-9

    T68 = T * 1.00024

    dens = a0 + \
           ((a1 + ((a2 + ((a3 + ((a4 + (a5 * T68)) * T68)) * T68)) * T68)) * T68)

    return dens


#def SWdensityFromCTD(depth_ctd, temp_ctd, sali_ctd, depths):
#    '''Calculate seawater density at CTD depth'''
#    import numpy
#
#    import utils_dtag
#
#    new_depth = range(max(depth_ctd))
#
#    temp = numpy.zeros(len(new_depth))
#    sali = numpy.zeros(len(new_depth))
#    temp[:] = numpy.nan
#    sali[:] = numpy.nan
#
#    temp[round(depth_ctd)] = temp_ctd
#    sali[round(depth_ctd)] = sali_ctd
#
#    # linear interpret between the NaNs
#    temp = utils_dtag.fixgaps(temp)
#    sali = utils_dtag.fixgaps(sali)
#    dens = sw_dens0(sali, temp)
#
#    ## TODO remove?
#    #plt.plot(dens, new_depth)
#    #ylabel('Depth (m)')
#    #xlabel('Density (kg/m^3)')
#
#    if max(depths[:, 6]) > max(depth_ctd):
#        new_depth2 = range(0, max(depths[:, 6]))
#        n_depth_diff = len(new_depth2)-len(new_depth)
#        sali = numpy.hstack([sali, sali[-1] * numpy.ones(n_depth_diff)])
#        temp = numpy.hstack([temp, temp[-1] * numpy.ones(n_depth_diff)])
#
#        dens = sw_dens0(sali, temp)
#
#        ## TODO remove?
#        #plt.plot(dens, new_depth2)
#        #ylabel('Depth (m)')
#        #xlabel('Density (kg/m^3)')
#
#    depth_CTD = numpy.copy(new_depth2)
#    SWdensity = numpy.copy(dens)
#
#    return SWdensity, depth_CTD
