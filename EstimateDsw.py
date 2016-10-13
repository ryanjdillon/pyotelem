
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
