
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


def add_bodydensity_to_experiments(path_field, path_isotope):
    import numpy
    import pandas

    import utils_seal_physio

    # Load experiments and convert datetimes to datetime
    field = pandas.read_csv(path_field, comment='#')
    field['date'] = pandas.to_datetime(field['date'])

    # Remove rows without an ID (experiments not to be used)
    field = field[~numpy.isnan(field['id'])]

    # Load isotope analysis and isotopemetric data, skip first 4 rows
    isotope = pandas.read_csv(path_isotope, comment='#')

    # Get percent body compositions, including density - what we want
    perc_comps = utils_seal_physio.lip2dens(isotope['fat_perc'])
    isotope['density_kgm3'] = perc_comps['density']*1000

    # List of columns to add to experiments from isotope-isotope data
    cols = ['mass_kg', 'length_cm', 'girth_cm','water_l', 'water_perc', 'fat_kg',
            'fat_perc', 'protein_kg', 'protein_perc', 'density_kgm3']

    # Create new columns in experiment dataframe
    for col in cols:
        field[col] = numpy.nan
    field['total_dens'] = numpy.nan

    # Add data from isotope-morpho dataframe to experiments dataframe
    for i in range(len(field)):
        idx = int(field['tritium_id'].iloc[i])
        midx = numpy.where(isotope['id'] == idx)[0][0]
        field.loc[i, cols] = isotope.ix[midx,cols]

        # Cacluate total density with modification, buoyant forces
        total_dens = apply_mods(field['mass_kg'][i],
                                                    field['density_kgm3'][i],
                                                    field['mod_type'][i],
                                                    field['n_mods'][i],
                                                    length=None, girth=None)
        field.loc[i, 'total_dens'] = total_dens


    return field, isotope


def make_field_isotope():
    import os

    from rjdtools import yaml_tools

    paths = yaml_tools.read_yaml('./cfg_paths.yaml')

    path_root = paths['root']
    path_csv = paths['csv']

    fname_field_csv    = 'field_experiments.csv'
    fname_isotope_csv = 'isotope_experiments.csv'

    fname_field_p      = 'field_experiments.p'
    fname_isotope_p   = 'isotope_experiments.p'

    path_field = os.path.join(path_root, path_csv, fname_field_csv)
    path_isotope = os.path.join(path_root, path_csv, fname_isotope_csv)

    field  field, isotope = add_bodydensity_to_experiments(path_field, path_isotope)

    field.to_pickle(os.path.join(path_root, path_csv, fname_field_p))
    isotope.to_pickle(os.path.join(path_root, path_csv, fname_isotope_p))

    return field, isotope



def filter_sgls(n_samples, exp_ind, sgls, max_pitch, min_depth,
        max_depth_delta, min_speed, max_speed, max_speed_delta):
    '''Create mask filtering only glides matching criterea'''
    import numpy

    from bodycondition import utils

    # Defined experiment indices
    mask_exp = (sgls['start_idx'] >= exp_ind[0]) & \
               (sgls['stop_idx'] <= exp_ind[-1])

    # Found within a dive
    mask_divid = ~numpy.isnan(sgls['dive_id'].astype(float))

    # Uniformity in phase (dive direction)
    mask_phase = (sgls['dive_phase'] == 'descent') | \
                 (sgls['dive_phase'] == 'ascent')

    # Depth change and minimum depth constraints
    mask_depth = (sgls['total_depth_change'] < max_depth_delta) & \
                 (sgls['total_depth_change'] > min_depth)

    # Pitch angle constraint
    mask_deg = (sgls['mean_pitch'] <  max_pitch) & \
               (sgls['mean_pitch'] > -max_pitch)

    # Speed constraints
    mask_speed = (sgls['mean_speed'] > min_speed) & \
                 (sgls['mean_speed'] < max_speed) & \
                 (sgls['total_speed_change'] < max_speed_delta)

    # Concatenate masks
    mask_sgls = mask_divid & mask_phase & mask_exp & \
                mask_deg    & mask_depth & mask_speed

    # Extract glide start/stop indices within above constraints
    start_ind = sgls[mask_sgls]['start_idx'].values
    stop_ind  = sgls[mask_sgls]['stop_idx'].values

    # Create mask for all data from valid start/stop indices
    mask_data_sgl = utils.mask_from_noncontiguous_indices(n_samples,
                                                          start_ind,
                                                          stop_ind)
    # Catch error with no matching subglides
    num_valid_sgls = len(numpy.where(mask_sgls)[0])
    if num_valid_sgls == 0:
        raise SystemError('No sublides found meeting filter criteria')

    return mask_data_sgl, mask_sgls


def get_subdir(path, cfg):
    import os

    from bodycondition import utils

    def match_subdir(path, cfg):
        import numpy

        n_subdirs = 0
        for d in os.listdir(path):
            if os.path.isdir(os.path.join(path, d)):
                n_subdirs += 1

        if n_subdirs == 0:
            raise SystemError('No data subdirectories in {}'.format(path))

        params = utils.parse_subdir(path)
        mask = numpy.zeros(n_subdirs, dtype=bool)

        # Evalute directory params against configuration params
        # Set directory mask to True where all parameters are matching
        for i in range(len(params)):
            match = list()
            for key, val in cfg.items():
                if params[key].iloc[i] == val:
                    match.append(True)
                else:
                    match.append(False)
            mask[i] = all(match)

        idx = numpy.where(mask)[0]
        if idx.size > 1:
            raise SystemError('More than one matching directory found')
        else:
            idx = idx[0]
            return params['name'].iloc[idx]


    # TODO this requires that each exp have same paramter values as in
    # cfg dict (i.e. cfg_ann and cfg_mcmc yaml)
    subdir_glide = match_subdir(path, cfg['glides'])

    path = os.path.join(path, subdir_glide)
    subdir_sgl   = match_subdir(path, cfg['sgls'])

    path = os.path.join(path, subdir_sgl)
    subdir_filt  = match_subdir(path, cfg['filter'])

    return os.path.join(subdir_glide, subdir_sgl, subdir_filt)


def compile_experiments(path_root, path_glide, cfg, fname_sgls,
        fname_mask_sgls, manual_selection=True):
    '''Compile data from experiments into three dataframes for MCMC input'''
    import numpy
    import os
    import pandas

    import utils
    from rjdtools import yaml_tools

    # List of paths to process
    path_exps = list()

    # Empty lists for appending IDs of each experiment
    exp_ids    = list()
    animal_ids = list()
    tag_ids    = list()

    print('''
          ┌----------------------------------------------------------------┐
          | Compiling glide analysis output to single file for model input |
          └----------------------------------------------------------------┘
          ''')

    # Iterate through experiment directories in glide analysis path
    first_iter = True

    # Generate list of possible paths to process in glide directory
    glide_paths_found = False
    for path_exp in os.listdir(os.path.join(path_root, path_glide)):
        path_data_glide = os.path.join(path_root, path_glide, path_exp)
        if os.path.isdir(path_data_glide):
            path_exps.append(path_exp)
            glid_paths_found = True

    # Throw exception if no data found in glide path
    if not glide_paths_found:
        raise SystemError('No glide paths found, check input directories '
                          'for errors\n'
                          'path_root: {}\n'
                          'path_glide: {}\n'.format(path_root, path_glide))

    # Run manual input data path selection, else process all present paths
    path_exps = sorted(path_exps)
    if manual_selection:
        msg = 'path numbers to compile to single dataset.\n'
        process_ind = utils.get_dir_indices(msg, path_exps)
    else:
        process_ind = range(len(path_exps))

    # Process user selected paths
    for i in process_ind:
        path_exp = path_exps[i]

        # Concatenate data path
        path_data_glide = os.path.join(path_root, path_glide, path_exp)
        path_subdir = get_subdir(path_data_glide, cfg)
        path_data_glide = os.path.join(path_data_glide, path_subdir)

        print('Processing {}'.format(path_exp))

        # Get experiment/animal ID from directory name
        exp_id    = path_exp
        tag_id    = exp_id.split('_')[2]
        animal_id = exp_id.split('_')[3]

        # Append experiment/animal id to list for `exps` df creation
        exp_ids.append(exp_id)
        animal_ids.append(animal_id)
        tag_ids.append(tag_id)

        # Read sgls dataframe, filter out only desired columns
        path_sgls = os.path.join(path_data_glide, fname_sgls)
        sgls_exp  = pandas.read_pickle(path_sgls)

        # Filter with saved mask meeting criteria
        # TODO pass filter routine as parameter to make more general
        path_mask_sgls = os.path.join(path_data_glide, fname_mask_sgls)
        mask_sgls      = numpy.load(path_mask_sgls)
        sgls_exp       = sgls_exp[mask_sgls]

        # Get unique dives in which all subglides occur
        dive_ids_exp = numpy.unique(sgls_exp['dive_id'][:])
        dives_exp = pandas.DataFrame(index=range(len(dive_ids_exp)))
        dives_exp['dive_id'] = dive_ids_exp
        # TODO read lung volume from file, or assign value here

        # Add exp_id/animal_id fields
        sgls_exp  = __add_ids_to_df(sgls_exp, exp_id)
        dives_exp = __add_ids_to_df(dives_exp, exp_id)

        # Append experiment sgl array to array with all exps to analyze
        if first_iter is True:
            first_iter = False
            sgls_all   = sgls_exp
            dives_all  = dives_exp
        else:
            sgls_all  = pandas.concat([sgls_all, sgls_exp], ignore_index = True)
            dives_all = pandas.concat([dives_all, dives_exp], ignore_index = True)

    # Create experiments dataframe
    exps_all = pandas.DataFrame(index=range(len(exp_ids)))
    exps_all = __add_ids_to_df(exps_all, exp_ids, animal_id=animal_ids,
                               tag_id=tag_ids)

    return exps_all, sgls_all, dives_all


def __add_ids_to_df(df, exp_id, animal_id=None, tag_id=None):
    '''Add columns to dataframe with experiment ID and animal ID

    if list of ids passed, must be equal to number of rows in `df`
    '''

    df['exp_id'] = exp_id

    # Add parameter if passed
    if animal_id is not None:
        df['animal_id']  = animal_id

    if tag_id is not None:
        df['tag_id']  = tag_id

    return df


def create_ann_inputs(path_root, path_acc, path_glide, path_ann, path_csv, fname_field_p,
        fname_sgls, fname_mask_sgls, sgl_cols, manual_selection=True):
    '''Compile all experiment data for ann model input'''
    import numpy
    import os
    import pandas

    from rjdtools import yaml_tools

    def insert_field_col_to_sgls(sgls, field):
        '''Insert bodycondition from nearest date in field to sgls dataframes'''
        import numpy

        col_name = 'total_dens'

        # Create empty column for body condition target values
        sgls = sgls.assign(**{col_name:numpy.full(len(sgls), numpy.nan)})

        exp_ids = numpy.unique(sgls['exp_id'].values)

        for exp_id in exp_ids:
            # TODO if using buoyancy, calculate with local seawater density

            mask_sgl = sgls['exp_id'] == exp_id
            mask_field = field['exp_id'] == exp_id

            try:
                value = field.ix[mask_field, 'total_dens'].values[0]
                sgls.ix[mask_sgl, col_name] = value
            except:
                raise SystemError('{} has no associated entries in the body '
                                  'composition dataframe'.format(exp_id))
        return sgls

    cfg_analysis = yaml_tools.read_yaml('./cfg_ann.yaml')

    # Compile subglide inputs for all experiments
    exps_all, sgls_all, dives_all = compile_experiments(path_root,
                                                        path_glide,
                                                        cfg_analysis['data'],
                                                        fname_sgls,
                                                        fname_mask_sgls)

    # Read body condition data
    path_field = os.path.join(path_root, path_csv, fname_field_p)
    field = pandas.read_pickle(path_field)

    # TODO could move this to `utils_glide`
    # Add integer dive_phase column
    des = sgls_all['dive_phase'] == 'descent'
    asc = sgls_all['dive_phase'] == 'ascent'

    sgls_all['dive_phase_int'] = 0
    sgls_all.ix[des, 'dive_phase_int'] = -1
    sgls_all.ix[asc, 'dive_phase_int'] = 1
    sgls_all.ix[~des&~asc, 'dive_phase_int'] = 0

    # Extract only columns useful for ann
    sgls = sgls_all[sgl_cols]

    # Add column with body condition target values to `sgls`
    sgls = insert_field_col_to_sgls(sgls, field)

    # Save output
    sgls.to_pickle(os.path.join(path_root, path_ann, 'sgls_all.p'))
    #exps_all.to_pickle(os.path.join(path_root, path_mcmc, 'exps_all.p'))
    #dives_all.to_pickle(os.path.join(path_root, path_mcmc, 'dives_all.p'))

    return exps_all, sgls, dives_all


def create_mcmc_inputs(path_root, path_glide, path_mcmc, fname_sgls,
        fname_mask_sgls, sgl_cols, manual_selection=True):
    '''Add MCMC distribution fields to each MCMC input dataframe'''
    import os
    import numpy

    from rjdtools import yaml_tools

    # TODO could create filter routine here to pass to compiler, pass arguments
    # for each input configuration, to generate inputs for model

    cfg_analysis = yaml_tools.read_yaml('./cfg_ann.yaml')

    # Compile subglide inputs for all experiments
    exps_all, sgls_all, dives_all = compile_experiments(path_root,
                                                        path_glide,
                                                        cfg_analysis['data'],
                                                        fname_sgls,
                                                        fname_mask_sgls)

    # Desired columns to extract from subglide analysis output
    sgls_all = sgls_all[sgl_cols]

    # Add for fields MCMC analysis output
    exp_new  = ['CdAm',     'CdAm_shape',     'CdAm_rate',
                'bdensity', 'bdensity_shape', 'bdensity_rate']
    sgl_new  = ['a', 'a_mu', 'a_tau']
    dive_new = ['v_air', 'v_air_shape', 'v_air_rate']

    exps_all  = __add_fields(exps_all, exp_new, numpy.nan)
    sgls_all  = __add_fields(sgls_all, sgl_new, numpy.nan)
    dives_all = __add_fields(dives_all, dive_new, numpy.nan)

    # Save output
    exps_all.to_pickle(os.path.join(path_root, path_mcmc, 'exps_all.p'))
    sgls_all.to_pickle(os.path.join(path_root, path_mcmc, 'sgls_all.p'))
    dives_all.to_pickle(os.path.join(path_root, path_mcmc, 'dives_all.p'))

    return exps_all, sgls_all, dives_all


def __add_fields(df, key_list, fill_value):
    '''Create new fields in dataframe filled with fill value'''
    for key in key_list:
        df[key] = fill_value
    return df


def make_model_inputs():
    from rjdtools import yaml_tools

    paths      = yaml_tools.read_yaml('./cfg_paths.yaml')
    path_root  = paths['root']
    path_acc   = paths['acc']
    path_glide = paths['glide']
    path_mcmc  = paths['mcmc']
    path_ann   = paths['ann']
    path_csv    = paths['csv']

    fname_sgls      = 'data_sgls.p'
    fname_mask_sgls = 'mask_sgls_filt.p'

    # Compile processed subglide data for MCMC model
    sgl_cols = ['exp_id', 'dive_id', 'mean_speed', 'mean_depth',
                'mean_sin_pitch', 'mean_swdensity', 'mean_a', 'SE_speed_vs_time']

    mcmc_exps, mcmc_sgls, mcmc_dives = create_mcmc_inputs(path_root,
                                                          path_glide,
                                                          path_mcmc,
                                                          fname_sgls,
                                                          fname_mask_sgls,
                                                          sgl_cols)

    # Compile processed subglide data for ANN model
    sgl_cols = ['exp_id', 'mean_speed', 'total_depth_change',
                'mean_sin_pitch', 'mean_swdensity', 'SE_speed_vs_time']

    fname_field_p = 'field_experiments.p'
    ann_exps, ann_sgls, ann_dives = create_ann_inputs(path_root,
                                                      path_acc,
                                                      path_glide,
                                                      path_ann,
                                                      path_csv,
                                                      fname_field_p,
                                                      fname_sgls,
                                                      fname_mask_sgls,
                                                      sgl_cols)

    return mcmc_exps, mcmc_sgls, mcmc_dives, ann_exps, ann_sgls, ann_dives


if __name__ == '__main__':

    mcmc_exps, mcmc_sgls, mcmc_dives, ann_exps, ann_sgls, ann_dives = make_model_inputs()
    field, isotope = make_field_isotope()
