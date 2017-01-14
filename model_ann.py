
def load_handwriting(feature_type, label_type, valid_frac):
    '''Load and randomly sample data to `train`, `validation`, `test` sets'''
    import pandas

    filename = './handwriting.csv'

    df = pandas.read_csv(filename, header=None)

    # Sample into train and validation sets
    df_train  = df.sample(frac=valid_frac)
    idx_train = df.index.isin(df_train.index)
    df_test   = df.loc[~idx_train]

    # Split valid to valid & test sets
    # http://stats.stackexchange.com/a/19051/16938
    valid_split = len(df_test)/2

    # Extract to numpy arrays
    train_array  = (df_train.ix[:,:61].values).astype(feature_type)
    train_labels = (df_train.ix[:,62].values).astype(label_type)

    valid_array  = (df_test.ix[:valid_split,:61].values).astype(feature_type)
    valid_labels = (df_test.ix[:valid_split,62].values).astype(label_type)

    test_array  = (df_test.ix[valid_split:,:61].values).astype(feature_type)
    test_labels = (df_test.ix[valid_split:,62].values).astype(label_type)

    # Make into tuple (features, label)
    train = train_array, train_labels
    valid = valid_array, valid_labels
    test = test_array, test_labels

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
    net = theanets.Classifier([n_features, config['hidden_nodes'],
                               n_targets])

    # Train the model using SGD with momentum.
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

    # Print confusion matrices for train and test
    valid_matrix = get_confusion_matrices(best_net, train, test)

    return tune_results, test_accuracy, valid_matrix


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

    # Print confusion matrices for train and test
    valid_matrix = get_confusion_matrices(best_net, train, test)

    return dataset_results, test_accuracy, valid_matrix


# TODO differs from net.score() result
#def calc_accuracy(valid_matrix):
#    '''Cacluate accuracy from confusion matrix results'''
#
#    accuracy = dict()
#    tot_true = 0
#    for i in range(matrix.shape[0]):
#        n_true = valid_matrix[i,i]
#        tot_i = sum(valid_matrix[:,i])
#        tot_true += n_true
#        #nF = sum(valid_matrix[:,i][:i])+sum(valid_matrix[:,i][(i+1):])
#        accuracy[i] = float(n_true)/float(tot_i)
#
#    accuracy['tot'] = float(tot_true)/float(sum(sum(valid_matrix)))
#
#    return accuracy


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

    # TODO make output directory, named dt, containing ann_cfg copy

    # TODO plots?

    # TODO add starttime, finishtime, git/versions
    from collections import OrderedDict
    import datetime
    import climate
    import os

    from rjdtools import yaml_tools

    climate.enable_default_logging()

    # Input
    debug = True
    ann_cfg_fname = 'cfg_ann.yaml'
    ann_cfg = yaml_tools.read_yaml(ann_cfg_fname)
    ann_path = './'

    train, valid, test = load_handwriting('f','i', 0.8)

    # Load data with random selection for validation
    # TODO automate from list of parameters to use as features, targets
    n_features = 62
    n_targets = 10

    # Output filenames
    tune_fname = 'results_tuning.p'
    dataset_size_fname = 'results_dataset_size.p'

    # Define output directory and creat if does not exist
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = 'theanets_{}'.format(now)

    out_path = os.path.join(ann_path, results_path)
    os.makedirs(out_path, exist_ok=True)

    # Define parameter set to tune neural net with
    if debug is True:
        tune_params = ann_cfg['debug']
    else:
        tune_params = ann_cfg['full']

    # Get all dict of all configuration permutations of params in `tune_params`
    configs = get_configs(tune_params)

    # Cycle through configurations storing configuration, net in `tune_results`
    tune_results, tune_accuracy, valid_matrix = tune_net(train, valid, test,
                                                         configs, n_features,
                                                         n_targets)

    # Get neural net configuration with best accuracy
    best_config = get_best(tune_results, 'config')

    # Get new randomly sorted and subsetted datasets to test effect of dataset_size
    # i.e. - a dataset with the first `subset_fraction` of samples.
    subset_fractions = [0.4, 0.7, 1.0]
    dataset_results, data_accuracy, valid_matrix = test_dataset_size(best_config,
                                                                  train, valid,
                                                                  test,
                                                                  subset_fractions)

    print('Test data accuracy (Configuration tuning): {}'.format(tune_accuracy))
    print('Test data accuracy (Datasize test):        {}'.format(data_accuracy))

    # Save results and configuration to output directory
    yaml_tools.write_yaml(ann_cfg, os.path.join(out_path, ann_cfg_fname))
    tune_results.to_pickle(os.path.join(out_path, tune_fname))
    dataset_results.to_pickle(os.path.join(out_path, dataset_size_fname))
