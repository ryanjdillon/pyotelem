
def SWdensityFromCTD(SA, t, p, potential=False):
    '''Calculate seawater density at CTD depth

    Args
    ----
    SA: ndarray
        Absolute salinity, g/kg
    t: ndarray
        In-situ temperature (ITS-90), degrees C
    p: ndarray
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    rho: ndarray
        Seawater density, in-situ or potential, kg/m^3
    '''
    import numpy
    import gsw

    CT = gsw.CT_from_t(SA, t, p)

    # Calculate potential density (0 bar) instead of in-situ
    if potential:
        p = numpy.zeros(len(SA))

    return gsw.rho(SA, CT, p)


def interp_S_t(S, t, z, z_new, p=None):
    ''' Linearly interpolate CTD S, t, and p (optional) from `z` to `z_new`.

    Args
    ----
    S: ndarray
        CTD salinities
    t: ndarray
        CTD temperatures
    z: ndarray
        CTD Depths, must be a strictly increasing or decreasing 1-D array, and
        its length must match the last dimension of `S` and `t`.
    z_new: ndarray
        Depth to interpolate `S`, `t`, and `p` to. May be a scalar or a sequence.
    p: ndarray (optional)
        CTD pressures

    Returns
    -------
    S_i: ndarray
        Interpolated salinities
    t_i: ndarray
        Interpolated temperatures
    p_i: ndarray
        Interpolated pressures. `None` returned if `p` not passed.

    Note
    ----
    It is assumed, but not checked, that `S`, `t`, and `z` are all plain
    ndarrays, not masked arrays or other sequences.

    Out-of-range values of `z_new`, and `nan` in `S` and `t`, yield
    corresponding `numpy.nan` in the output.

    This method is adapted from the the legacy `python-gsw` package, where
    their basic algorithm is from scipy.interpolate.
    '''
    import numpy

    # Create array-like query depth if single value passed
    isscalar = False
    if not numpy.iterable(z_new):
        isscalar = True
        z_new = [z_new]

    # Type cast to numpy array
    z_new = numpy.asarray(z_new)

    # Determine if depth direction is inverted
    inverted = False
    if z[1] - z[0] < 0:
        inverted = True
        z = z[::-1]
        S = S[..., ::-1]
        t = t[..., ::-1]
        if p is not None:
            p = p[..., ::-1]

    # Ensure query depths are ordered
    if (numpy.diff(z) <= 0).any():
        raise ValueError("z must be strictly increasing or decreasing")

    # Find z indices where z_new elements can insert with maintained order
    hi = numpy.searchsorted(z, z_new)

    # Replaces indices below/above with 1 or len(z)-1
    hi = hi.clip(1, len(z) - 1).astype(int)
    lo = hi - 1

    # Get CTD depths above and below query depths
    z_lo = z[lo]
    z_hi = z[hi]
    S_lo = S[lo]
    S_hi = S[hi]
    t_lo = t[lo]
    t_hi = t[hi]

    # Calculate distance ratio between CTD depths
    z_ratio = (z_new - z_lo) / (z_hi - z_lo)

    # Interpolate salinity and temperature with delta and ratio
    S_i = S_lo + (S_hi - S_lo) * z_ratio
    t_i = t_lo + (t_hi - t_lo) * z_ratio
    if p is None:
        p_i = None
    else:
        p_i = p[lo] + (p[hi] - p[lo]) * z_ratio

    # Invert interp values if passed depths inverted
    if inverted:
        S_i = S_i[..., ::-1]
        t_i = t_i[..., ::-1]
        if p is not None:
            p_i = p_i[..., ::-1]

    # Replace values not within CTD sample range with `nan`s
    outside = (z_new < z.min()) | (z_new > z.max())
    if numpy.any(outside):
        S_i[..., outside] = numpy.nan
        t_i[..., outside] = numpy.nan
        if p is not None:
            p_i[..., outside] = numpy.nan

    # Return single interp values if single query depth passed
    if isscalar:
        S_i = S_i[0]
        t_i = t_i[0]
        if p is not None:
            p_i = p_i[0]

    return S_i, t_i, p_i


#def estimate_seawater_density(depths, SA, t, p, duplicates='last'):
#    '''Estimate seawater density
#
#    depths: ndarray
#        Depths at which seawater density is to be estimated
#    depth_ctd:
#        Depths of CTD samples
#    temp_ctd:
#        CTD temperature values
#    sali_ctd:
#        CTD salinity values
#    duplicates: str
#        String indicating method to use when duplicate depths or pressures are
#        found in `depth_ctd`. 'first' will use the temperature and salinity
#        values from the first duplicate, and 'last' will use the values from
#        the last duplicate (Default 'last').
#
#    Returns
#    -------
#    tsd: pandas.DataFrame
#        Dataframe with temperature, salinity and depth at regular
#        whole integer depth or pressure intervals.
#    '''
#    import pandas
#
#    import utils
#
#    # Create empty data frame incremented by whole meters to max CTD depth
#    columns = ['temperature', 'salinity', 'density']
#    n_samples = numpy.ceil(depth_ctd.max()).astype(int)
#
#    #tsd = pandas.DataFrame(index=range(n_samples), columns=columns)
#    sa_i = numpy.zeros(n_samples, dtype=float)
#    t_i = numpy.zeros(n_samples, dtype=float)
#    p_i = numpy.zeros(n_samples, dtype=float)
#
#    # Assign temperature and salinity for each rounded ctd depth
#    depths = depth_ctd.round().astype(int)
#    for d in numpy.unique(depths):
#        # Use last or first occurance of depth/value pairs
#        if duplicates == 'last':
#            idx = numpy.where(depths == d)[0][-1]
#        elif duplicates == 'first':
#            idx = numpy.where(depths == d)[0][0]
#
#        # Fill temp and salinity at measured depths, rounded to whole meter
#        #tsd['temperature'][d] = temp_ctd[idx]
#        #tsd['salinity'][d]    = sali_ctd[idx]
#        sa_i[d] = sali_ctd[idx]
#        t_i[d] = temp_ctd[idx]
#        p_i[d] = pres_ctd[idx]
#
#    # Linearly interpolate temperature and salinity measurements
#    #tsd = tsd.astype(float)
#    tsd.interpolate('linear', inplace=True)
#    for a in [si, t_i, p_i]:
#        scipy.
#
#    tsd = SWdensityFromCTD(depth_ctd, temp_ctd, sali_ctd)
#
#    # TODO handle case where depth greater than CTD max, return last value/NaN
#    densities = tsd['density'][depths.round().astype(int)]
#
#    return tsd, densities

