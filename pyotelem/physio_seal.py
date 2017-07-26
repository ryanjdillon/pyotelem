
def lung_capacity(mass):
    '''Caclulate lung capacity

    Args
    ----
    mass: float
        Mass of animal

    Return
    ------
    volume: float
        Lung volume of animal

    References
    ----------
    Kooyman, G.L., Sinnett, E.E., 1979. Mechanical properties of the harbor
    porpoise lung, Phocoena phocoena. Respir Physiol 36, 287–300.
    '''
    return 0.135*(mass**0.92)


def calc_CdAm(farea, mass):
    '''Calculate drag term

    Args
    ----
    farea: float
        Frontal area of animal
    mass: float
        Total animal mass

    Returns
    -------
    CdAm: float
        Friction term of hydrodynamic equation

    References
    ----------
    Miller, P.J.O., Johnson, M.P., Tyack, P.L., Terray, E.A., 2004. Swimming
    gaits, passive drag and buoyancy of diving sperm whales Physeter
    macrocephalus. Journal of Experimental Biology 207, 1953–1967.
    doi:10.1242/jeb.00993
    '''
    Cd = 1.06
    return Cd*area/mass


def bodycomp(mass, tbw, method='reilly', simulate=False, n_rand=1000):
    '''Create dataframe with derived body composition values

    Args
    ----
    mass: ndarray
        Mass of the seal (kg)
    tbw: ndarray
        Total body water (kg)
    method: str
        name of method used to derive composition values
    simulate: bool
        switch for generating values with random noise
    n_rand: int
        number of density values to simulate

    Returns
    -------
    field: pandas.Dataframe
        dataframe containing columns for each body composition value

    References
    ----------
    Reilly, J.J., Fedak, M.A., 1990. Measurement of the body composition of
    living gray seals by hydrogen isotope dilution. Journal of Applied
    Physiology 69, 885–891.

    Gales, R., Renouf, D., Noseworthy, E., 1994. Body composition of harp
    seals. Canadian journal of zoology 72, 545–551.
    '''
    import numpy
    import pandas

    if len(mass) != len(tbw):
        raise SystemError('`mass` and `tbw` arrays must be the same length')

    bc = pandas.DataFrame(index=range(len(mass)))

    rnorm = lambda n, mu, sigma: numpy.random.normal(mu, sigma, n)

    if method == 'reilly':
        if simulate is True:
            bc['ptbw'] = 100 * (tbw / mass)
            bc['ptbf'] = 105.1 - (1.47 * bc['ptbw']) + rnorm(n_rand, 0, 1.1)
            bc['ptbp'] = (0.42 * bc['ptbw']) - 4.75 + rnorm(n_rand, 0, 0.8)
            bc['tbf'] = mass * (bc['ptbf'] / 100)
            bc['tbp'] = mass * (bc['ptbp'] / 100)
            bc['tba'] = 0.1 - (0.008 * mass) + \
                               (0.05 * tbw) + rnorm(0, 0.3, n_rand)
            bc['tbge'] = (40.8 * mass) - (48.5 * tbw) - \
                          0.4 + rnorm(0, 17.2, n_rand)
        else:
            bc['ptbw'] = 100 * (tbw / mass)
            bc['ptbf'] = 105.1 - (1.47 * bc['ptbw'])
            bc['ptbp'] = (0.42 * bc['ptbw']) - 4.75
            bc['tbf'] = mass * (bc['ptbf'] / 100)
            bc['tbp'] = mass * (bc['ptbp'] / 100)
            bc['tba'] = 0.1 - (0.008 * mass) + (0.05 * tbw)
            bc['tbge'] = (40.8 * mass) - (48.5 * tbw) - 0.4
    else:
        bc['ptbw'] = 100 * (tbw / mass)
        bc['tbf'] = mass - (1.37 * tbw)
        bc['tbp'] = 0.27 * (mass - bc['tbf'])
        bc['tbge'] = (40.8 * mass) - (48.5 * tbw) - 0.4
        bc['ptbf'] = 100 * (bc['tbf'] / mass)
        bc['ptbp'] = 100 * (bc['tbp'] / mass)

    return bc


def perc_bc_from_lipid(p_lipid):
    '''Calculate body composition component percentages based on % lipid

    Args
    ----
    p_lipid: ndarray
        Array of percent lipid values from which to calculate body composition

    Returns
    -------
    perc_comps: pandas.Dataframe
        Dataframe of percent composition values from percent lipids
    '''
    import pandas

    p_comps = pandas.DataFrame(index=range(len(p_lipid)))

    p_comps['perc_lipid']   = p_lipid
    p_comps['perc_water']   = 71.4966 - (0.6802721 * p_lipid)
    p_comps['perc_protien'] = (0.42 * p_comps['perc_water']) - 4.75
    p_comps['perc_ash']     = 100 - (p_lipid + p_comps['perc_water'] + \
                                     p_comps['perc_protien'])

    return p_comps


def water_from_lipid_protien(lipid, protein):
    '''Calculate total body water from total lipid and protein

    Args
    ----
    lipid: float or ndarray
        Mass of lipid content in animal
    protein: float or ndarray
        Mass of protein content in animal

    Returns
    -------
    water: float or ndarray
        Mass of water content in animal

    References
    ----------
    Reilly, J.J., Fedak, M.A., 1990. Measurement of the body composition of
    living gray seals by hydrogen isotope dilution. Journal of Applied
    Physiology 69, 885–891.
    '''
    return -4.408148e-16+(2.828348*protein) + (1.278273e-01*lipid)


def lip2dens(p_lipid, lipid_dens=0.9007, prot_dens=1.34, water_dens=0.994,
        a_dens=2.3):
    '''Derive tissue density from lipids

    Args
    ----
    p_lipid: float
        Percent lipid of body composition
    lipid_dens: float
        Density of lipid in animal (Default 0.9007 g/cm^3)
    prot_dens: float
        Density of protien in animal (Default 1.34 g/cm^3)
    water_dens: float
        Density of water in animal (Default 0.994 g/cm^3)
    a_dens: float
        Density of ash in animal (Default 2.3 g/cm^3)

    Returns
    -------
    p_comps: pandas.DataFrame
        Composition of body tissue components by percent

    References
    ----------
    Biuw, M., 2003. Blubber and buoyancy: monitoring the body condition of
    free-ranging seals using simple dive characteristics. Journal of
    Experimental Biology 206, 3405–3423. doi:10.1242/jeb.00583
    '''

    p_comps = perc_bc_from_lipid(p_lipid)

    p_comps['density'] = (lipid_dens * (0.01 * p_comps['perc_lipid'])) + \
                         (prot_dens  * (0.01 * p_comps['perc_protien'])) + \
                         (water_dens * (0.01 * p_comps['perc_water'])) + \
                         (a_dens     * (0.01 * p_comps['perc_ash']))
    return p_comps


def dens2lip(seal_dens, lipid_dens=0.9007, prot_dens=1.34, water_dens=0.994,
        a_dens=2.3):
    '''Get percent composition of animal from body density

    Args
    ----
    seal_dens: ndarray
        An array of seal densities (g/cm^3). The calculations only yield valid
        percents with densities between 0.888-1.123 with other parameters left
        as defaults.
    lipid_dens: float
        Density of lipid content in the animal (g/cm^3)
    prot_dens: float
        Density of protein content in the animal (g/cm^3)
    water_dens: float
        Density of water content in the animal (g/cm^3)
    a_dens: float
        Density of ash content in the animal (g/cm^3)

    Returns
    -------
    p_all: pandas.DataFrame
        Dataframe of components of body composition

    References
    ----------
    Biuw, M., 2003. Blubber and buoyancy: monitoring the body condition of
    free-ranging seals using simple dive characteristics. Journal of
    Experimental Biology 206, 3405–3423. doi:10.1242/jeb.00583
    '''
    import numpy

    # Convert any passed scalars to array-like and typecast to `numpy.array`
    if not numpy.iterable(seal_density):
        seal_density = [seal_density]
    seal_dens = numpy.array(seal_dens)

    ad_numerat =  -3.2248 * a_dens
    pd_numerat = -25.2786 * prot_dens
    wd_numerat = -71.4966 * water_dens

    ad_denom = -0.034  * a_dens
    pd_denom = -0.2857 * prot_dens
    wd_denom = -0.6803 * water_dens

    p_lipid = ((100 * seal_dens) + ad_numerat + pd_numerat + wd_numerat) / \
              (lipid_dens + ad_denom + pd_denom + wd_denom)

    p_all = lip2dens(p_lipid)
    p_all = p_all[['perc_water', 'perc_protien', 'perc_ash']]

    p_all['density'] = seal_dens
    p_all['perc_lipid'] = p_lipid

    return p_all


def buoyant_force(seal_dens, vol, sw_dens=1.028):
    '''Cacluate the buoyant force of an object in seawater

    Args
    ----
    seal_dens: float
        Density of the animal (g/cm^3)
    vol:
        Volume of the animal (cm^3)
    sw_dens: float
        Density of seawater (Default 1.028 g/cm^3)

    Returns
    -------
    Fb: float
        Buoyant force of animal
    '''
    # TODO review corrections with martin
    #g = 0.00980665 # gravity (km/s^2)
    g = 9.80665 # gravity (m/s^2)
    # 1000*(g/cm^3 - g/cm^3) * (1000 * cm^3) * km/s^2 != N (kg*m/s^2)
    # should be (1e-6 * vol), and g in m/s^2
    # kg/m^3 * (1e-6 * cm^3) * m/s^2 = N (kg*m/s^2)
    return (1000 * (sw_dens - dens)) * (1e-6 * vol) * g


def diff_speed(sw_dens=1.028, seal_dens=1.053, seal_length=300, seal_girth=200,
        Cd=0.09):
    '''Calculate terminal velocity of animal with a body size

    Args
    ----
    sw_dens: float
        Density of seawater (g/cm^3)
    seal_dens: float
        Density of animal (g/cm^3)
    seal_length: float
        Length of animal (cm)
    seal_girth: float
        Girth of animal (cm)
    Cd: float
        Drag coefficient of object in fluid, unitless

    Attributes
    ----------
    surf: float
        Surface area of animal (cm^2)
    vol: float
        Volume of animal (cm^3)
    Fb: float
        Buoyant force (N)
    x: float
        Inner term for of `Vt` calculation

    Returns
    -------
    Vt: float
        Terminal velocity of animal with given body dimensions (m/s).

    References
    ----------
    Biuw, M., 2003. Blubber and buoyancy: monitoring the body condition of
    free-ranging seals using simple dive characteristics. Journal of
    Experimental Biology 206, 3405–3423. doi:10.1242/jeb.00583

    Vogel, S., 1994. Life in Moving Fluids: The Physical Biology of Flow.
    Princeton University Press.
    '''
    import numpy

    surf, vol = surf_vol(seal_length, seal_girth)

    Fb = buoyant_force(seal_dens, vol, sw_dens)

    x = 2 * (Fb/(Cd * sw_dens * (surf*1000)))

    if x >= 0:
        Vt = numpy.sqrt(x)
    else:
        Vt = -numpy.sqrt(-x)

    return Vt


# TODO en? review with Martin
def lip2en(BM, perc_lipid):
    '''Percent lipid composition to percent energy stores

    Args
    ----
    BM: float or ndarray
        Body mass of animal
    perc_lipid: float or ndarray
        Percent lipids of animal's body composition

    Returns
    -------
    en: float or ndarray
        Total energy stores

    '''
    PTBW = 71.4966 - (0.6802721*perc_lipid)
    return (40.8*BM) - (48.5*(0.01*PTBW*BM)) - 0.4


def surf_vol(length, girth):
    '''Calculate the surface volume of an animal from its length and girth

    Args
    ----
    length: float or ndarray
        Length of animal (m)
    girth: float or ndarray
        Girth of animal (m)

    Returns
    -------
    surf:
        Surface area of animal (m^2)
    vol: float or ndarray
        Volume of animal (m^3)
    '''
    import numpy

    a_r   = 0.01 * girth / (2 * numpy.pi)
    stl_l = 0.01 * length
    c_r   = stl_l / 2
    e    = numpy.sqrt(1-(a_r**2/c_r**2))

    surf = ((2*numpy.pi * a_r**2) + \
           (2*numpy.pi * ((a_r * c_r)/e)) * 1/(numpy.sin(e)))

    vol  = (((4/3) * numpy.pi)*(a_r**2) * c_r)

    return surf, vol


def calc_seal_volume(mass_kg, dens_kgm3, length=None, girth=None):
    '''Calculate an animal's volume from mass and density or length and girth

    Args
    ----
    mass_kg: float or ndarray
        Mass of animal (kg)
    dens_kgm3: float or ndarray
        Density of animal (kg/m^3)
    length: float or None
        Length of animal. Default `None` (m)
    girth: float or None
        Girth of animal. Default `None` (m)

    Returns
    -------
    vol_kgm3: float or ndarray
        Volume of animal (m^3)
    '''
    if (length is not None) and (girth is not None):
        _, seal_vol = surf_vol(length, girth)
    else:
        seal_vol = mass_kg / dens_kgm3

    return seal_vol
