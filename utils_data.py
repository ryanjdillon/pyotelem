
def filter_sgls(n_samples, exp_ind, sgls, max_pitch, min_depth,
        max_depth_delta, min_speed, max_speed, max_speed_delta):
    '''Create mask filtering only glides matching criterea'''
    import numpy

    from bodycondition import utils

    # Defined experiment indices
    exp_mask = (sgls['start_idx'] >= exp_ind[0]) & \
               (sgls['stop_idx'] <= exp_ind[-1])

    # Found within a dive
    diveid_mask = ~numpy.isnan(sgls['dive_id'].astype(float))

    # Uniformity in phase (dive direction)
    phase_mask = (sgls['dive_phase'] == 'descent') | \
                 (sgls['dive_phase'] == 'ascent')

    # Depth change and minimum depth constraints
    depth_mask = (sgls['total_depth_change'] < max_depth_delta) & \
                 (sgls['total_depth_change'] > min_depth)

    # Pitch angle constraint
    deg_mask = (sgls['mean_pitch'] <  max_pitch) & \
               (sgls['mean_pitch'] > -max_pitch)

    # Speed constraints
    speed_mask = (sgls['mean_speed'] > min_speed) & \
                 (sgls['mean_speed'] < max_speed) & \
                 (sgls['total_speed_change'] < max_speed_delta)

    # Concatenate masks
    sgls_mask = diveid_mask & phase_mask & exp_mask & \
                deg_mask    & depth_mask & speed_mask

    # Extract glide start/stop indices within above constraints
    start_ind = sgls[sgls_mask]['start_idx'].values
    stop_ind  = sgls[sgls_mask]['stop_idx'].values

    # Create mask for all data from valid start/stop indices
    data_sgl_mask = utils.mask_from_noncontiguous_indices(n_samples,
                                                          start_ind,
                                                          stop_ind)
    # Catch error with no matching subglides
    num_valid_sgls = len(numpy.where(sgls_mask)[0])
    if num_valid_sgls == 0:
        raise SystemError('No sublides found meeting filter criteria')

    return data_sgl_mask, sgls_mask


def compile_experiments(root_path, glide_path, manual_selection=True):
    '''Compile data from experiments into three dataframes for MCMC input'''
    import numpy
    import os
    import pandas

    import utils

    sgls_filename = 'glide_sgls.p'
    sgls_mask_filename = 'glide_sgls_mask.npy'

    # List of paths to process
    exp_paths = list()

    # Empty lists for appending IDs of each experiment
    exp_ids    = list()
    animal_ids = list()
    tag_ids    = list()

    print('\nCompiling glide analysis output to single files for model input.\n')

    # Iterate through experiment directories in glide analysis path
    first_iter = True

    # Generate list of possible paths to process in glide directory
    glide_data_paths_found = False
    for exp_path in os.listdir(os.path.join(root_path, glide_path)):
        glide_data_path = os.path.join(root_path, glide_path, exp_path)
        if os.path.isdir(glide_data_path):
            exp_paths.append(exp_path)
            glide_data_paths_found = True

    # Throw exception if no data found in glide path
    if not glide_data_paths_found:
        raise SystemError('No glide paths found, check input directories '
                          'for errors\n'
                          'root_path: {}\n'
                          'glide_path: {}\n'.format(root_path, glide_path))

    # Run manual input data path selection, else process all present paths
    if manual_selection:
        msg = 'path numbers to compile to single dataset.\n'
        process_ind = utils.get_dir_indices(msg, exp_paths)
    else:
        process_ind = range(len(exp_paths))

    # Process user selected paths
    for i in process_ind:
        exp_path = exp_paths[i]

        print('Processing {}'.format(exp_path))

        # Get experiment/animal ID from directory name
        exp_id    = exp_path
        tag_id    = exp_id.split('_')[2]
        animal_id = exp_id.split('_')[3]

        # Append experiment/animal id to list for `exps` df creation
        exp_ids.append(exp_id)
        animal_ids.append(animal_id)
        tag_ids.append(tag_id)

        # Read sgls dataframe, filter out only desired columns
        sgls_path = os.path.join(glide_data_path, sgls_filename)
        sgls_exp  = pandas.read_pickle(sgls_path)

        # Filter with saved mask meeting criteria
        # TODO pass filter routine as parameter to make more general
        sgls_mask_path = os.path.join(glide_data_path, sgls_mask_filename)
        sgls_mask      = numpy.load(sgls_mask_path)
        sgls_exp       = sgls_exp[sgls_mask]

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


def read_bodycondition(filename):
    '''Reads body condition csv or pandas pickle and add density column'''
    import os
    import pandas

    import utils_morphometrics

    # TODO make columns uniform for body composition

    # Read to pandas dataframe
    if filename.endswith('.csv'):
        bc = pandas.read_csv(filename)
    elif filename.endswith('.p'):
        bc = pandas.read_pickle(filename)
    else:
        raise SystemError('Filename must be either .csv or pandas .p '
                          'pickle format.')

    # Remove any extra whitespace in titles or animal str column
    bc.rename(columns= lambda x: x.strip())
    bc['animal'] = list(map(str.strip, bc['animal']))

    # Get percent body compositions, including density - what we want
    perc_comps = utils_morphometrics.lip2dens(bc['fat_perc'])
    bc['density'] = perc_comps['density']

    # Write as pandas pickle file to same directory
    bc.to_pickle(os.path.splitext(filename)[0]+'.p')

    return bc


def create_ann_inputs(root_path, acc_path, glide_path, ann_path, bc_path, bc_filename,
        sgl_cols, manual_selection=True):
    '''Compile all experiment data for ann model input'''
    import numpy
    import os
    import pandas

    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    def add_col(df, col_name, values):
        '''Add column `col_name` dataframe with values'''
        return df.assign(**{col_name:values})

    def density_mod(mod, mod_n):
        '''Calculate density modification given type and number of blocks'''
        # Density of experimental blocks (kg/L)
        # dims: 15x4x3cm, neutral blocks ~density of seawater
        mod_dens = {'float':0.189, 'weight':3.55, 'neutral':0, 'control':0}
        return mod_dens[mod]*mod_n

    def insert_bc_col_to_sgls(root_path, acc_path, sgls, bc, col_name):
        '''Insert bodycondition from nearest date in bc to sgls dataframes'''
        import datetime
        import numpy
        import tqdm

        from rjdtools import yaml_tools

        # Create empty column for body condition target values
        sgls = add_col(sgls, col_name, numpy.full(len(sgls), numpy.nan))

        exp_ids = numpy.unique(sgls['exp_id'].values)

        for i in tqdm.tqdm(range(len(exp_ids))):
            exp_id = exp_ids[i]
            meta_path = os.path.join(root_path, acc_path, exp_id, 'meta.yaml')
            meta = yaml_tools.read_yaml(meta_path)
            mod   = meta['mod']   # type of block
            mod_n = meta['mod_n'] # number of blocks

            # Get datetime from subglide `exp_id`
            date_str = (exp_id.split('_'))[0]
            sgl_dt = datetime.datetime.strptime(date_str, '%Y%m%d')

            # Get `bc` mask from subglide's animal ID in `exp_id`
            animal = (exp_id.split('_')[3]).lower()
            bc_animal_mask = bc['animal'] == animal

            # Make sure an entry for this animal exists in body condition data
            if len(numpy.where(bc_animal_mask)) == 0:
                raise KeyError('Animal ID `{}` not found in body condition '
                               'dataframe `bc`.'.format(animal))

            # find date in `bc` (filtered by animal) nearest sgl date
            bc_dt = nearest(bc['date'][bc_animal_mask], sgl_dt)

            # Get `bc` index position of nearest datetime
            bc_idx = numpy.argmax(bc['date'] == bc_dt)

            exp_mask = sgls['exp_id'] == exp_id
            sgls[exp_mask] = bc[col_name].iloc[bc_idx] + density_mod(mod, mod_n)

            # Write body condition value to `sgl` subglide row
            # add density modification from attached blocks
            # TODO review

        return sgls

    # Compile subglide inputs for all experiments
    exps_all, sgls_all, dives_all = compile_experiments(root_path, glide_path)

    # Read body condition data
    bc_file_path = os.path.join(root_path, bc_path, bc_filename)
    bc = read_bodycondition(bc_file_path)

    # TODO review columns useful for ann
    sgls = sgls_all[sgl_cols]

    # Add column with body condition target values to `sgls`
    sgls = insert_bc_col_to_sgls(root_path, acc_path, sgls, bc, 'density')

    # TODO normalize input fields

    # Save output
    sgls_all.to_pickle(os.path.join(root_path, ann_path, 'sgls_all.p'))
    #exps_all.to_pickle(os.path.join(root_path, mcmc_path, 'exps_all.p'))
    #dives_all.to_pickle(os.path.join(root_path, mcmc_path, 'dives_all.p'))

    return sgls

def create_mcmc_inputs(root_path, glide_path, mcmc_path, sgl_cols,
        manual_selection=True):
    '''Add MCMC distribution fields to each MCMC input dataframe'''
    import os
    import numpy

    # TODO could create filter routine here to pass to compiler, pass arguments
    # for each input configuration, to generate inputs for model

    # Compile subglide inputs for all experiments
    exps_all, sgls_all, dives_all = compile_experiments(root_path, glide_path,
            manual_selection=manual_selection)

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
    exps_all.to_pickle(os.path.join(root_path, mcmc_path, 'exps_all.p'))
    sgls_all.to_pickle(os.path.join(root_path, mcmc_path, 'sgls_all.p'))
    dives_all.to_pickle(os.path.join(root_path, mcmc_path, 'dives_all.p'))

    return exps_all, sgls_all, dives_all


def __add_fields(df, key_list, fill_value):
    '''Create new fields in dataframe filled with fill value'''
    for key in key_list:
        df[key] = fill_value
    return df


if __name__ == '__main__':
    from rjdtools import yaml_tools

    paths      = yaml_tools.read_yaml('./iopaths.yaml')
    root_path  = paths['root']
    acc_path   = paths['acc']
    glide_path = paths['glide']
    mcmc_path  = paths['mcmc']
    ann_path   = paths['ann']
    bc_path    = paths['bodycondition']

    # Compile processed subglide data for MCMC model
    sgl_cols = ['exp_id', 'dive_id', 'mean_speed', 'mean_depth',
                'mean_sin_pitch', 'mean_swdensity', 'mean_a', 'SE_speed_vs_time']

    mcmc_exps, mcmc_sgls, mcmc_dives = create_mcmc_inputs(root_path,
                                                          glide_path,
                                                          mcmc_path,
                                                          sgl_cols)

    # Compile processed subglide data for ANN model
    sgl_cols = ['exp_id', 'mean_speed', 'total_depth_change',
                'mean_sin_pitch', 'mean_swdensity', 'SE_speed_vs_time']

    bc_filename = 'bc_no-tag_skinny_yellow.p'
    ann_exps, ann_sgls, ann_dives = create_ann_inputs(root_path,
                                                      acc_path,
                                                      glide_path,
                                                      ann_path,
                                                      bc_path,
                                                      bc_filename,
                                                      sgl_cols)
