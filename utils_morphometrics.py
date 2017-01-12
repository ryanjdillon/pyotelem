
def lung_capacity(mass):
    '''Caclulate lung capacity Kooyman and Sinnett (1981)'''
    return 0.135*(mass**0.92)


def calc_CdAm(farea, mass):
    '''Calculate drag term

    Args
    ----
    farea: float
        frontal area of animal
    mass: float
        total animal mass

    Returns
    -------
    CdAm: float
        friction term of hydrodynamic equation

    Miller et al. 2004: Cd set to 1.06, prolate spheroid, fineness ratio 5.0
    '''
    Cd = 1.06
    return Cd*area/mass

