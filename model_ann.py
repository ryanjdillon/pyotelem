
def n_hidden(n_input, n_output, n_train_samples, alpha):
    # http://stats.stackexchange.com/a/136542/16938
    # Alpha is scaling factor between 2-10
    n_hidden = n_samples/(alpha*(n_input+n_output))
    return n_hidden

def split_data(df, feature_cols, target_col, valid_frac):
    '''Load and randomly sample data to `train`, `validation`, `test` sets'''
    import numpy
    import pandas
    from sklearn.preprocessing import normalize

    # Sample into train and validation sets
    df_train  = df.sample(frac=valid_frac)
    idx_train = df.index.isin(df_train.index)
    df_test   = df.loc[~idx_train]

    # Split valid to valid & test sets
    # http://stats.stackexchange.com/a/19051/16938
    valid_split = len(df_test)//2

    # Extract to numpy arrays - typecast to float32
    train_array  = (df_train[feature_cols].values)
    train_labels = (df_train[target_col].values)

    valid_array  = (df_test[feature_cols][:valid_split].values)
    valid_labels = (df_test[target_col][:valid_split].values)

    test_array  = (df_test[feature_cols][valid_split:].values)
    test_labels = (df_test[target_col][valid_split:].values)

    # Normalize inputs
    X_train = normalize(train_array, norm='l2', axis=1).astype('f4')
    X_valid = normalize(valid_array, norm='l2', axis=1).astype('f4')
    X_test  = normalize(test_array, norm='l2', axis=1).astype('f4')

    # Normalize outputs
    y_train = (normalize(train_labels, norm='l2', axis=1)).astype('f4')
    y_valid = (normalize(valid_labels, norm='l2', axis=1)).astype('f4')
    y_test  = (normalize(test_labels, norm='l2', axis=1)).astype('f4')

    # Make into tuple (features, label)
    # Pivot 1-D target value arrays to match 0dim of X
    train = X_train, y_train.T
    valid = X_valid, y_valid.T
    test  =  X_test,  y_test.T

    return train, valid, test


def get_confusion_matrices(net, train, valid):
    '''Print and return an sklearn confusion matrix from the input net'''
    from sklearn.metrics import confusion_matrix

    # Show confusion matrices on the training/validation splits.
    for label, (X, y) in (('training:', train), ('validation:', valid)):
        print(label)
        print(confusion_matrix(y, net.predict(X)))

    valid_matrix = confusion_matrix(y, net.predict(X))

    return valid_matrix


def get_configs(tune_params):
    '''Generate list of all possible configuration dicts from tuning params'''
    import itertools

    #tune_idxs = list()
    #for key in tune_params.keys():
    #    tune_idxs.append(range(len(tune_params[key])))

    # Create list of configuration permutations
    config_list = list(itertools.product(*tune_params.values()))

    configs = list()
    for l in config_list:
        configs.append(dict(zip(tune_params.keys(), l)))

    return configs


def create_algorithm(train, valid, config, n_features, n_targets):
    '''Configure and train a theanets neural network'''
    import theanets

    # Build neural net with defined configuration
    #net = theanets.Classifier([n_features, config['hidden_nodes'],
    #                           n_targets])
    # User 'mse' as loss function
    net = theanets.Regressor(layers=[n_features, config['hidden_nodes'], n_targets])

    # Train the model using SGD with momentum.
    # user net.itertrain() for return monitors to plot
    net.train(train, valid, algo=config['algorithm'],
                            hidden_l1=config['l1'],
                            weight_l2=config['l2'],
                            learning_rate=1e-4,
                            momentum=0.9
                            )

    # Classify features against label/target value to get accuracy
    # where `valid` is a tuple with validation (features, label)
    accuracy = net.score(valid[0], valid[1])

    return net, accuracy


def get_best(results, key):
    '''Return results column 'key''s value from model with best accuracy'''
    best_idx = results['accuracy'].idxmax()
    return results[key][best_idx]


def tune_net(train, valid, test, configs, n_features, n_targets):
    '''Train nets with varying configurations and `validation` set

    The determined best configuration is then used to find the resulting
    accuracy with the `test` dataset
    '''
    import numpy
    import pandas

    tune_cols = ['config', 'net', 'accuracy']
    tune_results = pandas.DataFrame(index=range(len(configs)),
                                    columns=tune_cols, dtype=object)
    #tune_results = numpy.zeros((len(configs),3), dtype=object)

    for i in range(len(configs)):

        net, accuracy = create_algorithm(train, valid, configs[i], n_features,
                                         n_targets)

        tune_results['config'][i]   = configs[i]
        tune_results['net'][i]      = net
        tune_results['accuracy'][i] = accuracy

    # Get neural net with best accuracy
    best_net = get_best(tune_results, 'net')

    # Classify features against label/target value to get accuracy
    # where `test` is a tuple with test (features, label)
    test_accuracy = best_net.score(test[0], test[1])
    print('tune test accuracy: {}'.format(test_accuracy))

    # Print confusion matrices for train and test
    #valid_matrix = get_confusion_matrices(best_net, train, test)

    return tune_results, test_accuracy#, valid_matrix


def truncate_data(data, frac):
    '''Reduce data rows to `frac` of original

    Args
    ----
    data: Tuple containing numpy array of feature data and labels
    frac: percetange of original data to return
    '''

    n = len(data[0])
    subset_frac = (data[0][:round(n*frac)], data[1][:round(n*frac)])

    return subset_frac


def test_dataset_size(best_config, train, valid, test, subset_fractions):
    '''Train nets with best configuration and varying dataset sizes'''
    import numpy
    import pandas

    # Make array for storing results
    data_cols = ['config', 'net', 'accuracy', 'subset_frac']
    dataset_results = pandas.DataFrame(index=range(len(subset_fractions)),
                                    columns=data_cols, dtype=object)

    # Generate net and save results for each data subset
    for i in range(len(subset_fractions)):

        # Trim data sets to `frac` of original
        train_frac = truncate_data(train, subset_fractions[i])
        valid_frac = truncate_data(valid, subset_fractions[i])
        test_frac  = truncate_data(test, subset_fractions[i])

        net, accuracy = create_algorithm(train, valid, best_config, n_features,
                                         n_targets)

        dataset_results['config'][i]      = best_config
        dataset_results['net'][i]         = net
        dataset_results['accuracy'][i]    = accuracy
        dataset_results['subset_frac'][i] = subset_fractions[i]

    # Get neural net with best accuracy
    best_net = get_best(dataset_results, 'net')

    # Classify features against label/target value to get accuracy
    # where `test` is a tuple with test (features, label)
    test_accuracy = best_net.score(test[0], test[1])
    print('dataset size test accuracy: {}'.format(test_accuracy))

    # Print confusion matrices for train and test
    #valid_matrix = get_confusion_matrices(best_net, train, test)

    return dataset_results, test_accuracy#, valid_matrix


if __name__ == '__main__':
    # The following loads the handwriting dataset, tunes different nets with
    # varying configuration, and then tests the effect of dataset size.

    # NOTE: the validation set is split into "validation" and "test" sets, the
    # first used for initial comparisons of various net configuration
    # accuracies and the second for a clean test set to get an true accuracy,
    # as reusing the "validation" set can cause the routine to overfit to the
    # validation set.

    # TODO check if results are still loadable as pandas instead of pkl
    #pickle.dump(results, open('results_data.pkl', 'wb'))

    # TODO make following script a routine, debug as option
    # convert npy arrays to pandas with column labels

    # TODO plots?

    # TODO add starttime, finishtime, git/versions
    from collections import OrderedDict
    import datetime
    import climate
    import numpy
    import os
    import theano

    import utils_data
    from rjdtools import yaml_tools

    climate.enable_default_logging()
    theano.config.compute_test_value = 'off'

    # Input
    debug = False # TODO remove
    ann_cfg_fname = 'cfg_ann.yaml'
    ann_cfg = yaml_tools.read_yaml(ann_cfg_fname)

    # Output filenames
    tune_fname = 'results_tuning.p'
    dataset_size_fname = 'results_dataset_size.p'

    # Define paths
    paths = yaml_tools.read_yaml('./iopaths.yaml')
    root_path  = paths['root']
    acc_path   = paths['acc']
    glide_path = paths['glide']
    ann_path   = paths['ann']
    bc_path    = paths['bodycondition']
    bc_filename = 'bc_no-tag_skinny_yellow.p'

    # Define output directory and creat if does not exist
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = 'theanets_{}'.format(now)
    out_path = os.path.join(root_path, ann_path, results_path)
    os.makedirs(out_path, exist_ok=True)

    print('Compile output from glides into ANN input')

    # Compile output from glides into single input dataframe
    sgl_cols = ['exp_id', 'mean_speed', 'total_depth_change',
                'mean_sin_pitch', 'mean_swdensity', 'SE_speed_vs_time']
    sgls = utils_data.create_ann_inputs(root_path, acc_path, glide_path,
                                        ann_path, bc_path, bc_filename,
                                        sgl_cols, manual_selection=True)

    # TODO review outcome of this
    sgls = sgls.dropna()

    # Dimensions of Input and output layers
    feature_cols = ['mean_speed', 'total_depth_change', 'mean_sin_pitch',
                    'mean_swdensity', 'SE_speed_vs_time']
    target_col = 'density'
    valid_frac = 0.8

    print('Split and normalize input/output data')

    # Split data with random selection for validation
    train, valid, test = split_data(sgls, feature_cols, target_col, valid_frac)

    # Define parameter set to tune neural net with
    if debug is True:
        tune_params = ann_cfg['debug']
    else:
        tune_params = ann_cfg['full']

    print('Tune netork configuration')

    # Get all dict of all configuration permutations of params in `tune_params`
    configs = get_configs(tune_params)

    # Cycle through configurations storing configuration, net in `tune_results`
    n_features = len(feature_cols)
    n_targets = len(numpy.unique(train[1]))
    print('features: {}'.format(n_features))
    print('targets: {}'.format(n_targets))
    n_targets = 1
    tune_results, tune_accuracy = tune_net(train, valid, test, configs,
                                           n_features, n_targets)

    # Get neural net configuration with best accuracy
    best_config = get_best(tune_results, 'config')

    print('Run percentage of datasize tests')

    # Get new randomly sorted and subsetted datasets to test effect of dataset_size
    # i.e. - a dataset with the first `subset_fraction` of samples.
    subset_fractions = [0.4, 0.7, 1.0]
    dataset_results, data_accuracy = test_dataset_size(best_config, train,
                                                       valid, test, subset_fractions)

    print('Test data accuracy (Configuration tuning): {}'.format(tune_accuracy))
    print('Test data accuracy (Datasize test):        {}'.format(data_accuracy))

    # Save results and configuration to output directory
    yaml_tools.write_yaml(ann_cfg, os.path.join(out_path, ann_cfg_fname))
    tune_results.to_pickle(os.path.join(out_path, tune_fname))
    dataset_results.to_pickle(os.path.join(out_path, dataset_size_fname))
