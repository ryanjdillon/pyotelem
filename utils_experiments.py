
def simulate_density(mass_kg=40, bd_start=1000, n_bd=101, block_start=1,
        n_blocks=8):
    '''Produce a range of body densities given an initial mass and density'''
    import numpy
    import pandas

    # Range of body densities to test
    types = ['weight', 'float']
    bd_range = bd_start+numpy.arange(0, n_bd)
    bodydensities = numpy.tile(numpy.repeat(bd_range, n_blocks), len(types))

    block_range = block_start+numpy.arange(0, n_blocks)
    blocks = numpy.tile(numpy.tile(block_range, n_bd), len(types))

    types = numpy.repeat(types, n_bd*n_blocks)

    columns = ['type', 'dens_kgm3', 'n_blocks', 'rho_mod', 'delta_rho']
    df = pandas.DataFrame(index=range(len(bodydensities)), columns=columns)

    for i in range(len(df)):
        print(i, df.index[i])
        df.loc[df.index[i], 'type'] = types[i]
        df.loc[df.index[i], 'dens_kgm3'] = bodydensities[i]
        df.loc[df.index[i], 'n_blocks'] = blocks[i]
        #seal_vol = calc_seal_volume(mass_kg, bodydensities[i])
        #rho_mod = calc_mod_density(mass_kg, seal_vol, blocks[i], t)
        rho_mod = calc_mod_density_kagari(mass_kg, bodydensities[i], blocks[i], t)
        df.loc[df.index[i], 'rho_mod'] = rho_mod
        df.loc[df.index[i], 'delta_ro'] = rho_mod - bodydensities[i]

    return df


def calc_mod_density(mass_kg, seal_vol, n_mods, mod_type):
    '''Cacluate the total density of the seal with modification blocks'''
    # Modifier block attributes
    mod_vol  = 0.15 * 0.04 * 0.03 # Length x Width x Height (m^3)
    mod_dens = {'weight': 3556.0, 'float': 744.0}
    mod_mass = {'weight': 0.640,  'float': 0.134}

    # Calculate combined density
    total_mass = (mass_kg + (n_mods * mod_mass[mod_type]))
    total_vol  = (seal_vol + (n_mods * mod_vol))
    total_dens = total_mass / total_vol

    return total_dens


def calc_mod_density_kagari(mass_kg, dens_kgm3, n_blocks, mod_type):
    def mod_weight(mass_kg, dens_kgm3, n_weights):
        total_dens = ((mass_kg*1000 + 168*4 + 260*n_weights) /
                      (mass_kg*1000 / (dens_kgm3/1000) + 168*4) * 1000)
        return total_dens

    def mod_float(mass_kg, dens_kgm3, n_weights):
        total_dens = ((mass_kg*1000 + 168*(4-n_floats) + 35*n_floats) /
                      (mass_kg*1000 / (dens_kgm3/1000) + 168*4) * 1000)
        return total_dens

    if mod_type == 'weight':
        total_dens = mod_weight(mass_kg, dens_kgm3, n_blocks)
    elif mod_type == 'float':
        total_dens = mod_float(mass_kg, dens_kgm3, n_blocks)
    else:
        raise ValueError('mod_type must be "weight" or "float"')

    return total_dens


def apply_mods(mass_kg, dens_kgm3, mod_type, n_mods, length=None, girth=None,
        dsw_kgm3=1028.0):
    '''Estimate change in buoyancy with floats or weights

    Args
    ----
    mass_kg: float
        mass of the seal (kg)
    dens_kgm3: float
        mass of the seal (kg/m^3)
    mod_type: str
        Type of modication block for experiment (`weight` or `float`)
    n_mods: int
        number of modifying blocks attached
    length: float
        length of seal (m)
    girth: float
        girth of seal (m)

    Returns
    -------
    total_dens: float
        combined density of seal and attached blocks

    Notes
    -----
    Block attributes from correspondance with Martin, differ from Kagari's
    '''

    # Only update density if 'weight' or 'float' mod_type
    if (mod_type == 'weight') or (mod_type == 'float'):

        seal_vol = calc_seal_volume(mass_kg, dens_kgm3)
        total_dens = calc_mod_density(mass_kg, seal_vol, n_mods, mod_type)

    # Density of seal unchanged if not 'weight' or 'float' (i.e. 'control', 'neutral')
    else:
        total_dens = dens_kgm3

    return total_dens


def add_morpho_to_experiments(exp_file, morpho_file):
    import numpy
    import pandas

    import utils_morphometrics

    # Load experiments and convert datetimes to datetime
    exps = pandas.read_csv(exp_file, comment='#')
    exps['date'] = pandas.to_datetime(exps['date'])

    # Remove rows without an ID (experiments not to be used)
    exps = exps[~numpy.isnan(exps['id'])]

    # Load tritium analysis and morphometric data, skip first 4 rows
    morpho = pandas.read_csv(morpho_file, comment='#')

    # Get percent body compositions, including density - what we want
    perc_comps = utils_morphometrics.lip2dens(morpho['fat_perc'])
    morpho['density_kgm3'] = perc_comps['density']*1000

    # List of columns to add to experiments from tritium-morpho data
    cols = ['mass_kg', 'length_cm', 'girth_cm','water_l', 'water_perc', 'fat_kg',
            'fat_perc', 'protein_kg', 'protein_perc', 'density_kgm3']

    # Create new columns in experiment dataframe
    for col in cols:
        exps[col] = numpy.nan
    exps['total_dens'] = numpy.nan

    # Add data from tritium-morpho dataframe to experiments dataframe
    for i in range(len(exps)):
        idx = int(exps['tritium_id'].iloc[i])
        midx = numpy.where(morpho['id'] == idx)[0][0]
        exps.loc[i, cols] = morpho.ix[midx,cols]

        # Cacluate total density with modification, buoyant forces
        total_dens = apply_mods(exps['mass_kg'][i],
                                                    exps['density_kgm3'][i],
                                                    exps['mod_type'][i],
                                                    exps['n_mods'][i],
                                                    length=None, girth=None)
        exps.loc[i, 'total_dens'] = total_dens


    return exps, morpho


def make_exps_morpho():
    import os

    from rjdtools import yaml_tools

    paths = yaml_tools.read_yaml('./cfg_paths.yaml')

    root_path = paths['root']
    morpho_path = paths['bodycondition']

    fname_exp_csv    = 'coexist_experiments.csv'
    fname_morpho_csv = 'coexist_tritium-morphometrics.csv'

    fname_exp_p      = 'coexist_experiments.p'
    fname_morpho_p   = 'coexist_morphometrics.p'

    exp_file = os.path.join(root_path, morpho_path, fname_exp_csv)
    morpho_file = os.path.join(root_path, morpho_path, fname_morpho_csv)

    exps, morpho = add_morpho_to_experiments(exp_file, morpho_file)

    exps.to_pickle(os.path.join(root_path, morpho_path, fname_exp_p))
    morpho.to_pickle(os.path.join(root_path, morpho_path, fname_morpho_p))

    return exps, morpho


if __name__ == '__main__':
    make_exps_morpho()
