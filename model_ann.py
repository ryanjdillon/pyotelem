
def n_hidden(n_input, n_output, n_train_samples, alpha):
    # http://stats.stackexchange.com/a/136542/16938
    # Alpha is scaling factor between 2-10
    n_hidden = n_samples/(alpha*(n_input+n_output))
    return n_hidden


def split_data(df, feature_cols, target_col, valid_frac, n_classes):
    '''Load and randomly sample data to `train`, `validation`, `test` sets'''
    import numpy
    import pandas
    from sklearn.preprocessing import normalize

    # TODO add bin sizes to cfg

    # Bin outputs
    ymin =  df[target_col].min()
    ymax =  df[target_col].max()
    mod = (ymax - ymin)/n_classes/4
    bin_min = ymin - mod
    bin_max = ymax + mod
    bins = numpy.linspace(bin_min, bin_max, n_classes)
    df['y_binned'] = numpy.digitize(df[target_col], bins)

    # Sample into train and validation sets
    df_train  = df.sample(frac=valid_frac)
    idx_train = df.index.isin(df_train.index)
    df_test   = df.loc[~idx_train]

    # Split valid to valid & test sets
    # http://stats.stackexchange.com/a/19051/16938
    valid_split = len(df_test)//2

    # Extract to numpy arrays - typecast to float32
    train_array  = (df_train[feature_cols].values)
    train_labels = (df_train['y_binned'].values)

    valid_array  = (df_test[feature_cols][:valid_split].values)
    valid_labels = (df_test['y_binned'][:valid_split].values)

    test_array  = (df_test[feature_cols][valid_split:].values)
    test_labels = (df_test['y_binned'][valid_split:].values)

    # Normalize inputs
    X_train = normalize(train_array, norm='l2', axis=1).astype('f4')
    X_valid = normalize(valid_array, norm='l2', axis=1).astype('f4')
    X_test  = normalize(test_array, norm='l2', axis=1).astype('f4')

    ## Normalize outputs
    #y_train = (normalize(train_labels, norm='l2', axis=1)).astype('f4')
    #y_valid = (normalize(valid_labels, norm='l2', axis=1)).astype('f4')
    #y_test  = (normalize(test_labels, norm='l2', axis=1)).astype('f4')

    # Make into tuple (features, label)
    # Pivot 1-D target value arrays to match 0dim of X
    train = X_train, train_labels.astype('i4')
    valid = X_valid, valid_labels.astype('i4')
    test  = X_test, test_labels.astype('i4')

    return train, valid, test, bins


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
    net = theanets.Classifier([n_features, config['hidden_nodes'], n_targets])

    # User 'mse' as loss function
    #net = theanets.Regressor(layers=[n_features, config['hidden_nodes'], n_targets])

    # mini-batchs
    # http://sebastianruder.com/optimizing-gradient-descent/index.html#minibatchgradientdescent
    # https://github.com/lmjohns3/theanets/blob/master/scripts/theanets-char-rnn

    # TODO use other trainers (optimizers)
    # https://theanets.readthedocs.io/en/stable/api/trainers.html
    # http://sebastianruder.com/optimizing-gradient-descent/
    # NAG, ADADELTA, RMSProp

    # Input/hidden dropout
    # Input/hidden noise

    # Learning rate

    # Shuffling, Curriculum learning
    # http://sebastianruder.com/optimizing-gradient-descent/index.html#shufflingandcurriculumlearning

    # Batch normalization?
    # http://sebastianruder.com/optimizing-gradient-descent/index.html#batchnormalization

    # Early stopping https://github.com/lmjohns3/theanets/issues/17

    # TODO figure out mini-batches, data callable
    # https://groups.google.com/forum/#!topic/theanets/LctHBDAKdH8
    #batch_size = 64

    #if not train_batches:
    #    train_batchs = batch_size
    #if not valid_batches:
    #    valid_batches = batch_size

    # SGD converges to minima/maxima faster with momentum
    # NAG, ADADELTA, RMSProp have equivalents of parameter specific momentum
    if config['algorithm'] is 'sgd':
        config['momentum'] = 0.9

    #train_loss = list()
    #valid_loss = list()
    #for train_monitor, valid_monitor in exp.itertrain(...):
    #    train_loss.append(train_monitor['loss'])
    #    valid_loss.append(valid_monitor['loss'])

    ## Plot loss during training
    #plt.plot(loss)
    #plt.show()

    # Train the model using SGD with momentum.
    # user net.itertrain() for return monitors to plot
    net.train(train,
              valid,
              algo=config['algorithm'],
              patience=config['patience'],
              min_improvement=config['min_improvement'],
              validate_every =config['validate_every'],
              #batch_size=batch_size,
              #train_batches=train_batches,
              #valid_batches=valid_batches,
              learning_rate=config['learning_rate'],
              momentum=config['momentum'],
              hidden_l1=config['hidden_l1'],
              weight_l2=config['weight_l2'],
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
    import time

    tune_cols = ['config', 'net', 'accuracy', 'train_time']
    results_tune = pandas.DataFrame(index=range(len(configs)),
                                    columns=tune_cols, dtype=object)
    #results_tune = numpy.zeros((len(configs),3), dtype=object)

    for i in range(len(configs)):

        t0 = time.time()

        net, accuracy = create_algorithm(train, valid, configs[i], n_features,
                                         n_targets)

        t1 = time.time()

        results_tune['config'][i]     = configs[i]
        results_tune['net'][i]        = net
        results_tune['accuracy'][i]   = accuracy
        results_tune['train_time'][i] = t1 - t0

    # Get neural net with best accuracy
    best_net = get_best(results_tune, 'net')

    # Classify features against label/target value to get accuracy
    # where `test` is a tuple with test (features, label)
    test_accuracy = best_net.score(test[0], test[1])
    print('tune test accuracy: {}'.format(test_accuracy))

    # Print confusion matrices for train and test
    valid_matrix = get_confusion_matrices(best_net, train, test)

    return results_tune, test_accuracy, valid_matrix


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


def test_dataset_size(best_config, train, valid, test, n_features, n_targets,
        subset_fractions):
    '''Train nets with best configuration and varying dataset sizes'''
    import numpy
    import pandas
    import time

    # Make array for storing results
    data_cols = ['config', 'net', 'accuracy', 'subset_frac', 'train_time']
    results_dataset = pandas.DataFrame(index=range(len(subset_fractions)),
                                    columns=data_cols, dtype=object)

    # Generate net and save results for each data subset
    for i in range(len(subset_fractions)):

        # Trim data sets to `frac` of original
        train_frac = truncate_data(train, subset_fractions[i])
        valid_frac = truncate_data(valid, subset_fractions[i])
        test_frac  = truncate_data(test, subset_fractions[i])

        t0 = time.time()

        net, accuracy = create_algorithm(train, valid, best_config, n_features,
                                         n_targets)
        t1 = time.time()

        results_dataset['config'][i]      = best_config
        results_dataset['net'][i]         = net
        results_dataset['accuracy'][i]    = accuracy
        results_dataset['subset_frac'][i] = subset_fractions[i]
        results_dataset['train_time'][i]  = t1 - t0

    # Get neural net with best accuracy
    best_net = get_best(results_dataset, 'net')

    # Classify features against label/target value to get accuracy
    # where `test` is a tuple with test (features, label)
    test_accuracy = best_net.score(test[0], test[1])

    print('dataset size test accuracy: {}'.format(test_accuracy))

    # Print confusion matrices for train and test
    valid_matrix = get_confusion_matrices(best_net, train, test)

    return results_dataset, test_accuracy, valid_matrix

#import click
#
#
#@click.command(help='Run ANN with parameters defined in cfg_ann.yaml')
#
#@click.option('--cfg-paths-path', prompt=True, default='./cfg_paths.yaml',
#              help='Path to cfg_paths.yaml')
#
#@click.option('--cfg-ann-path', prompt=True, default='./cfg_ann.yaml',
#              help='Path to cfg_ann.yaml')
#
#@click.option('--debug', prompt=True, default=False, type=bool,
#              help='Use single permutation of tuning parameters')


def run(cfg_paths_path, cfg_ann_path, debug=False):
    '''
    Compile subglide data, tune network architecture and test dataset size

    Note
    ----
    The validation set is split into "validation" and "test" sets, the
    first used for initial comparisons of various net configuration
    accuracies and the second for a clean test set to get an true accuracy,
    as reusing the "validation" set can cause the routine to overfit to the
    validation set.
    '''

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

    # Environment settings - logging, Theano, load configuration, set paths
    #---------------------------------------------------------------------------
    climate.enable_default_logging()
    theano.config.compute_test_value = 'ignore'

    # Configuration settings
    cfg = yaml_tools.read_yaml(cfg_ann_path)
    if debug is True:
        for key in cfg['net_tuning'].keys():
            cfg['net_tuning'][key] = [cfg['net_tuning'][key][0],]

    # Define output directory and create if does not exist
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg['output']['results_path'] = 'theanets_{}'.format(now)

    # Define paths
    paths       = yaml_tools.read_yaml(cfg_paths_path)
    root_path   = paths['root']
    acc_path    = paths['acc']
    glide_path  = paths['glide']
    ann_path    = paths['ann']
    bc_path     = paths['bodycondition']
    bc_filename = 'bc_no-tag_skinny_yellow.p'


    # Compile, split, and normalize data
    #---------------------------------------------------------------------------
    print('Compile output from glides into ANN input')

    # Compile output from glides into single input dataframe
    exps, sgls, dives = utils_data.create_ann_inputs(root_path,
                                                     acc_path,
                                                     glide_path,
                                                     ann_path,
                                                     bc_path,
                                                     bc_filename,
                                                     cfg['data']['sgl_cols'],
                                                     manual_selection=True)
    # TODO review outcome of this
    # TODO add num. ascent/descent glides to cfg
    sgls = sgls.dropna()

    print('Split and normalize input/output data')
    # Split data with random selection for validation
    train, valid, test, bins = split_data(sgls,
                                          cfg['net_all']['feature_cols'],
                                          cfg['net_all']['target_col'],
                                          cfg['net_all']['valid_frac'],
                                          cfg['net_all']['n_classes'])


    # Tuning - find optimal network architecture
    #---------------------------------------------------------------------------
    print('Tune netork configuration')

    # Get all dict of all configuration permutations of params in `tune_params`
    configs = get_configs(cfg['net_tuning'])

    # Cycle through configurations storing configuration, net in `results_tune`
    n_features = len(cfg['net_all']['feature_cols'])
    n_targets = cfg['net_all']['n_classes']

    print('features: {}'.format(n_features))
    print('targets: {}'.format(n_targets))
    #n_targets = 1
    results_tune, tune_accuracy, tune_matrix = tune_net(train,
                                                        valid,
                                                        test,
                                                        configs,
                                                        n_features,
                                                        n_targets)

    # Get neural net configuration with best accuracy
    best_config = get_best(results_tune, 'config')


    # Test effect of dataset size
    #---------------------------------------------------------------------------
    print('Run percentage of datasize tests')

    # Get randomly sorted and subsetted datasets to test effect of dataset_size
    # i.e. - a dataset with the first `subset_fraction` of samples.
    subset_fractions = cfg['net_dataset_size']['subset_fractions']
    results_dataset, data_accuracy, data_matrix = test_dataset_size(best_config,
                                                                    train,
                                                                    valid,
                                                                    test,
                                                                    n_features,
                                                                    n_targets,
                                                                    subset_fractions)

    print('Test data accuracy (Configuration tuning): {}'.format(tune_accuracy))
    print('Test data accuracy (Datasize test):        {}'.format(data_accuracy))


    # Save results and configuration to output directory
    #---------------------------------------------------------------------------
    out_path = os.path.join(root_path, ann_path, cfg['output']['results_path'])
    os.makedirs(out_path, exist_ok=True)

    yaml_tools.write_yaml(cfg, os.path.join(out_path, cfg_ann_path))
    results_tune.to_pickle(os.path.join(out_path, cfg['output']['tune_fname']))
    results_dataset.to_pickle(os.path.join(out_path, cfg['output']['dataset_size_fname']))

    return cfg, train, valid, test, results_tune, results_dataset


if __name__ == '__main__':
    #cfg, results_tune, results_dataset = run()

    cfg_paths_path = './cfg_paths.yaml'
    cfg_ann_path   = './cfg_ann.yaml'
    debug = True
    cfg, train, valid, test, results_tune, results_dataset = run(cfg_paths_path,
                                                                 cfg_ann_path,
                                                                 debug=debug)
