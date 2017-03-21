
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
    glide_data_paths_found = False
    for path_exp in os.listdir(os.path.join(path_root, path_glide)):
        glide_data_path = os.path.join(path_root, path_glide, path_exp)
        if os.path.isdir(glide_data_path):
            path_exps.append(path_exp)
            glide_data_paths_found = True

    # Throw exception if no data found in glide path
    if not glide_data_paths_found:
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
        glide_data_path = os.path.join(path_root, path_glide, path_exp)
        subdir_path = get_subdir(glide_data_path, cfg)
        glide_data_path = os.path.join(glide_data_path, subdir_path)

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
        sgls_path = os.path.join(glide_data_path, fname_sgls)
        sgls_exp  = pandas.read_pickle(sgls_path)

        # Filter with saved mask meeting criteria
        # TODO pass filter routine as parameter to make more general
        sgls_mask_path = os.path.join(glide_data_path, fname_mask_sgls)
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


def create_ann_inputs(path_root, path_acc, path_glide, path_ann, path_bc, fname_bc,
        fname_sgls, fname_mask_sgls, sgl_cols, manual_selection=True):
    '''Compile all experiment data for ann model input'''
    import numpy
    import os
    import pandas

    from rjdtools import yaml_tools

    def insert_bc_col_to_sgls(sgls, bc):
        '''Insert bodycondition from nearest date in bc to sgls dataframes'''
        import numpy

        col_name = 'total_dens'

        # Create empty column for body condition target values
        sgls = sgls.assign(**{col_name:numpy.full(len(sgls), numpy.nan)})

        exp_ids = numpy.unique(sgls['exp_id'].values)

        for exp_id in exp_ids:
            # TODO if using buoyancy, calculate with local seawater density

            mask_sgl = sgls['exp_id'] == exp_id
            mask_bc = bc['exp_id'] == exp_id

            try:
                value = bc.ix[mask_bc, 'total_dens'].values[0]
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
    bc_file_path = os.path.join(path_root, path_bc, fname_bc)
    bc = pandas.read_pickle(bc_file_path)

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
    sgls = insert_bc_col_to_sgls(sgls, bc)

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


if __name__ == '__main__':
    from rjdtools import yaml_tools

    paths      = yaml_tools.read_yaml('./cfg_paths.yaml')
    path_root  = paths['root']
    path_acc   = paths['acc']
    path_glide = paths['glide']
    path_mcmc  = paths['mcmc']
    path_ann   = paths['ann']
    path_bc    = paths['bodycondition']

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

    fname_bc = 'coexist_experiments.p'
    ann_exps, ann_sgls, ann_dives = create_ann_inputs(path_root,
                                                      path_acc,
                                                      path_glide,
                                                      path_ann,
                                                      path_bc,
                                                      fname_bc,
                                                      fname_sgls,
                                                      fname_mask_sgls,
                                                      sgl_cols)
