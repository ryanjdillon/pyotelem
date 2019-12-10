
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
    elif method == 'gales':
        if simulate is True:
            raise ValueError('Random error simulation is currently only '
                'implemented for `method` `reilly`. `simulate` must be passed '
                'as `False` when using `method` `gales`.')
        else:
            bc['ptbw'] = 100 * (tbw / mass)
            bc['tbf'] = mass - (1.37 * tbw)
            bc['tbp'] = 0.27 * (mass - bc['tbf'])
            bc['tbge'] = (40.8 * mass) - (48.5 * tbw) - 0.4
            bc['ptbf'] = 100 * (bc['tbf'] / mass)
            bc['ptbp'] = 100 * (bc['tbp'] / mass)
    else:
        raise ValueError('`method` must be either `reilly` or `gales`, not '
            '`{}`'.format(method))

    return bc


def perc_bc_from_lipid(perc_lipid, perc_water=None):
    '''Calculate body composition component percentages based on % lipid

    Calculation of percent protein and percent ash are based on those presented
    in Reilly and Fedak (1990).

    Args
    ----
    perc_lipid: float or ndarray
        1D array of percent lipid values from which to calculate body composition
    perc_water: float or ndarray
        1D array of percent water values from which to calculate body
        composition (Default `None`). If no values are passed, calculations are
        performed with values from Biuw et al. (2003).

    Returns
    -------
    perc_water: float or ndarray
        1D array of percent water values
    perc_protein: float or ndarray
        1D array of percent protein values
    perc_ash: float or ndarray
        1D array of percent ash values

    References
    ----------
    Biuw, M., 2003. Blubber and buoyancy: monitoring the body condition of
    free-ranging seals using simple dive characteristics. Journal of
    Experimental Biology 206, 3405–3423. doi:10.1242/jeb.00583

    Reilly, J.J., Fedak, M.A., 1990. Measurement of the body composition of
    living gray seals by hydrogen isotope dilution. Journal of Applied
    Physiology 69, 885–891.
    '''
    import numpy

    # Cast iterables to numpy arrays
    if numpy.iterable(perc_lipid):
        perc_lipid = numpy.asarray(perc_lipid)
    if numpy.iterable(perc_water):
        perc_water = numpy.asarray(perc_water)

    if not perc_water:
        # TODO check where `perc_water` values come from
        perc_water   = 71.4966 - (0.6802721 * perc_lipid)

    perc_protein = (0.42 * perc_water) - 4.75
    perc_ash     = 100 - (perc_lipid + perc_water + perc_protein)

    return perc_water, perc_protein, perc_ash


def water_from_lipid_protein(lipid, protein):
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


def lip2dens(perc_lipid, dens_lipid=0.9007, dens_prot=1.34, dens_water=0.994,
        dens_ash=2.3):
    '''Derive tissue density from lipids

    The equation calculating animal density is from Biuw et al. (2003), and
    default values for component densities are from human studies collected in
    the book by Moore et al. (1963).

    Args
    ----
    perc_lipid: float or ndarray
        Percent lipid of body composition
    dens_lipid: float
        Density of lipid in animal (Default 0.9007 g/cm^3)
    dens_prot: float
        Density of protein in animal (Default 1.34 g/cm^3)
    dens_water: float
        Density of water in animal (Default 0.994 g/cm^3)
    dens_ash: float
        Density of ash in animal (Default 2.3 g/cm^3)

    Returns
    -------
    dens_gcm3: float or ndarray
        Density of seal calculated from percent compositions and densities of
        components from Moore et al. (1963)

    References
    ----------
    Biuw, M., 2003. Blubber and buoyancy: monitoring the body condition of
    free-ranging seals using simple dive characteristics. Journal of
    Experimental Biology 206, 3405–3423. doi:10.1242/jeb.00583

    Moore FD, Oleson KH, McMurrery JD, Parker HV, Ball MR, Boyden CM. The Body
    Cell Mass and Its Supporting Environment - The Composition in Health and
    Disease. Philadelphia: W.B. Saunders Company; 1963. 535 p.
    ISBN:0-7216-6480-6
    '''
    import numpy

    # Cast iterables to numpy array
    if numpy.iterable(perc_lipid):
        perc_lipid = numpy.asarray(perc_lipid)

    perc_water, perc_protein, perc_ash = perc_bc_from_lipid(perc_lipid)

    dens_gcm3 = (dens_lipid * (0.01 * perc_lipid)) + \
                (dens_prot  * (0.01 * perc_protein)) + \
                (dens_water * (0.01 * perc_water)) + \
                (dens_ash   * (0.01 * perc_ash))

    return dens_gcm3


def dens2lip(dens_gcm3, dens_lipid=0.9007, dens_prot=1.34, dens_water=0.994,
        dens_ash=2.3):
    '''Get percent composition of animal from body density

    The equation calculating animal density is from Biuw et al. (2003), and
    default values for component densities are from human studies collected in
    the book by Moore et al. (1963).

    Args
    ----
    dens_gcm3: float or ndarray
        An array of seal densities (g/cm^3). The calculations only yield valid
        percents with densities between 0.888-1.123 with other parameters left
        as defaults.
    dens_lipid: float
        Density of lipid content in the animal (g/cm^3)
    dens_prot: float
        Density of protein content in the animal (g/cm^3)
    dens_water: float
        Density of water content in the animal (g/cm^3)
    dens_ash: float
        Density of ash content in the animal (g/cm^3)

    Returns
    -------
    perc_all: pandas.DataFrame
        Dataframe of components of body composition

    References
    ----------
    Biuw, M., 2003. Blubber and buoyancy: monitoring the body condition of
    free-ranging seals using simple dive characteristics. Journal of
    Experimental Biology 206, 3405–3423. doi:10.1242/jeb.00583

    Moore FD, Oleson KH, McMurrery JD, Parker HV, Ball MR, Boyden CM. The Body
    Cell Mass and Its Supporting Environment - The Composition in Health and
    Disease. Philadelphia: W.B. Saunders Company; 1963. 535 p.
    ISBN:0-7216-6480-6
    '''
    import numpy

    # Cast iterables to numpy array
    if numpy.iterable(dens_gcm3):
        dens_gcm3 = numpy.asarray(dens_gcm3)

    # Numerators
    ad_num =  -3.2248 * dens_ash
    pd_num = -25.2786 * dens_prot
    wd_num = -71.4966 * dens_water

    # Denominators
    ad_den = -0.034  * dens_ash
    pd_den = -0.2857 * dens_prot
    wd_den = -0.6803 * dens_water

    perc_lipid = ((100 * dens_gcm3) + ad_num + pd_num + wd_num) / \
                        (dens_lipid + ad_den + pd_den + wd_den)

    return perc_lipid


def buoyant_force(dens_gcm3, vol, sw_dens=1.028):
    '''Cacluate the buoyant force of an object in seawater

    Args
    ----
    dens_gcm3: float
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


def diff_speed(sw_dens=1.028, dens_gcm3=1.053, seal_length=300, seal_girth=200,
        Cd=0.09):
    '''Calculate terminal velocity of animal with a body size

    Args
    ----
    sw_dens: float
        Density of seawater (g/cm^3)
    dens_gcm3: float
        Density of animal (g/cm^3)
    seal_length: float
        Length of animal (cm)
    seal_girth: float
        Girth of animal (cm)
    Cd: float
        Drag coefficient of object in fluid, unitless

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

    Fb = buoyant_force(dens_gcm3, vol, sw_dens)

    x = 2 * (Fb/(Cd * sw_dens * (surf*1000)))

    if x >= 0:
        Vt = numpy.sqrt(x)
    else:
        Vt = -numpy.sqrt(-x)

    return Vt


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
