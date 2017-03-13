
def add_morpho_to_experiments(exp_file, morph_file):
    import numpy
    import pandas

    import utils_morphometrics

    # Load experiments and convert datetimes to datetime
    exps = pandas.read_csv(exp_file, comment='#')
    exps['date'] = pandas.to_datetime(exps['date'])

    # Remove rows without an ID (experiments not to be used)
    exps = exps[~numpy.isnan(exps['id'])]

    # Load tritium analysis and morphometric data, skip first 4 rows
    morph = pandas.read_csv(morph_file, comment='#')

    # Get percent body compositions, including density - what we want
    perc_comps = utils_morphometrics.lip2dens(morph['fat_perc'])
    morph['density_kgm3'] = perc_comps['density']*1000

    # List of columns to add to experiments from tritium-morpho data
    cols = ['mass_kg', 'water_l', 'water_perc', 'fat_kg', 'fat_perc',
            'protein_kg', 'protein_perc', 'density_kgm3', 'mass_scale', 'length',
            'girth']

    # Create new columns in experiment dataframe
    for col in cols:
        exps[col] = numpy.nan
    exps['total_dens'] = numpy.nan

    # Add data from tritium-morpho dataframe to experiments dataframe
    for i in range(len(exps)):
        idx = int(exps['tritium_id'].iloc[i])
        exps.loc[i, cols] = morph.loc[idx, cols]

        # Cacluate total density with modification, buoyant forces
        total_dens = utils_morphometrics.apply_mods(exps['mass_kg'][i],
                                                    exps['density_kgm3'][i],
                                                    exps['mod_type'][i],
                                                    exps['n_mods'][i],
                                                    length=None, girth=None)
        exps.loc[i, 'total_dens'] = total_dens


    return exps


if __name__ == '__main__':
    import os

    from rjdtools import yaml_tools

    paths = yaml_tools.read_yaml('./cfg_paths.yaml')

    root_path = paths['root']
    morph_path = paths['bodycondition']

    exp_file = os.path.join(root_path, morph_path, 'coexist_experiments.csv')
    morph_file = os.path.join(root_path, morph_path, 'coexist_tritium-morphometrics.csv')

    exps = add_morpho_to_experiments(exp_file, morph_file)

    exps.to_pickle(os.path.join(root_path, morph_path, 'coexist_experiments.p'))
