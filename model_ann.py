
def n_hidden(n_input, n_output, n_train_samples, alpha):
    # http://stats.stackexchange.com/a/136542/16938
    # Alpha is scaling factor between 2-10
    n_hidden = n_samples/(alpha*(n_input+n_output))
    return n_hidden


def normalize_data(df, features, target, n_targets):
    '''Normalize features and target values

    Args
    ----
    n_targets: int
        Number of target targets (bins) to split `target` into

    Returns
    -------
    df: pandas.DataFrame
        Data frame containing normalized features and target columns
    bins: ndarray
        List of unique targets (bins) generated during data splitting

    Note
    ----
    This should be performed before splitting the dataset into train/validation
    sets.

    Could also consider using sklearn for normalization:

    from sklearn.preprocessing import normalize
    data_norm = normalize(data_array, norm='l2', axis=1).astype('f4')
    '''

    def mean_normalize(df, keys):
        return df[keys].div(df.sum(axis=0)[keys], axis=1)[keys]

    def unit_normalize(df, keys):
        data = df[keys].values
        #(df[keys] - df[keys].min())/(df[keys].max()-df[keys].min())
        return (data - data.min(axis=0))/(data.max(axis=0) - data.min(axis=0))

    def bin_column(df, target, n_targets):
        import numpy

        ymin =  df[target].min()
        ymax =  df[target].max()
        mod = (ymax - ymin)/n_targets/4
        bin_min = ymin - mod
        bin_max = ymax + mod
        bins = numpy.linspace(bin_min, bin_max, n_targets)

        return numpy.digitize(df[target], bins), bins

    # Normalize inputs
    df[features] = unit_normalize(df, features)

    # Bin outputs
    df['y_binned'], bins = bin_column(df, target, n_targets)

    return df, bins


def get_split_indexes(df, valid_frac):
    '''Get randomly selected row indices for train, validation, and test data
    Args
    ----
    df: pandas.DataFrame
        
    features: list
        List of string column names in `df` to be used as feature values
    target: str
        Name of the column in `df` to use as the target value
    valid_frac: float
        Fraction of dataset that should be reserved for validation/testing.
        This slice of dataframe `df` is then split in half into the validation
        and testing datasets

    Returns
    -------
    ind_train: pandas.index
        Index of values to be sliced for training
    ind_valid: pandas.index
        Index of values to be sliced for validation
    ind_test: pandas.index
        Index of values to be sliced for testing
    '''
    # Sample into train and validation sets
    ind_train = df.sample(frac=valid_frac).index
    mask_rest = df.index.isin(ind_train)
    df_rest = df.loc[~mask_rest]

    # Split valid to valid & test sets
    # http://stats.stackexchange.com/a/19051/16938
    ind_valid = df_rest.sample(frac=0.5).index
    ind_test  = df_rest[~df_rest.index.isin(ind_valid)].index

    return ind_train, ind_valid, ind_test


def create_datasets(df, ind_train, ind_valid, ind_test, features, target):
    '''

    Args
    ----
    df: pandas.DataFrame
        

    Returns
    -------
    train: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training
    valid: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training validation
    test: tuple (ndarray, ndarray)
        Tuple containing feature and target values for testing
    '''

    df_train = df.iloc[ind_train]
    df_valid = df.iloc[ind_valid]
    df_test  = df.iloc[ind_test]

    # Extract to numpy arrays - typecast to float32
    X_train  = (df_train[features].values).astype('f4')
    train_labels = (df_train['y_binned'].values).astype('f4')

    X_valid  = (df_test[features].values).astype('f4')
    valid_labels = (df_test['y_binned'].values).astype('f4')

    X_test  = (df_test[features].values).astype('f4')
    test_labels = (df_test['y_binned'].values).astype('f4')

    # Make into tuple (features, label)
    # Pivot 1-D target value arrays to match 0dim of X
    train = X_train, train_labels.astype('i4')
    valid = X_valid, valid_labels.astype('i4')
    test  = X_test, test_labels.astype('i4')

    return train, valid, test




def plot_confusion_matrix(cm, targets, normalize=False, title='', cmap=None):
    '''This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    Plotting routind modified from this code: https://goo.gl/kYHMxk
    '''
    import itertools
    import matplotlib.pyplot as plt
    import numpy

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    n_targets = len(targets)
    yticks = numpy.arange(n_targets)
    xticks = yticks #- (numpy.diff(yticks)[0]/3)
    plt.xticks(xticks, numpy.round(targets, 1), rotation=45)
    plt.yticks(yticks, numpy.round(targets,1))

    if cmap is None:
        cmap = plt.cm.Blues

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print('\n{}, normalized'.format(title))
    else:
        print('\n{}, without normalization'.format(title))

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return None


def get_confusion_matrices(net, train, valid, targets):
    '''Print and return an sklearn confusion matrix from the input net'''
    from collections import OrderedDict
    import numpy
    import sklearn.metrics

    # Filter targets to only those that were assigned to training values
    targets = targets[sorted(list(numpy.unique(train[1])))]

    # Show confusion matrices on the training/validation splits.
    cms = OrderedDict()
    for label, (X, y) in (('Training', train), ('Validation', valid)):
        title = '{} confusion matrix'.format(label)
        label = label.lower()
        cms[label] = sklearn.metrics.confusion_matrix(y, net.predict(X))
        plot_confusion_matrix(cms[label], targets, title=title)
    return cms


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


def print_dict_values(cfg):
    '''Print parameterization of input data to be analyzed'''

    labels = ['glides', 'sgls', 'filter']

    line = '-' * max([len(l) for l in labels]) * 4

    print('\nInput data configuration:')

    for label in labels:
        pad_front = ' '*((len(line) - len(label))//2)
        pad_back  = ' '*((len(line) - len(label))%2)
        print('\n' + pad_front + label.upper() + pad_back)
        print(line)
        space = str(max([len(key) for key in list(cfg[label])]))
        for key, value in cfg[label].items():
            print(('{:>'+space+'}: {:04.2f}').format(key, value))

        print(line)

    return None


def create_algorithm(train, valid, config, n_features, n_targets, plots=False):
    '''Configure and train a theanets neural network

    Args
    ----
    train: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training
    valid: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training validation
    test: tuple (ndarray, ndarray)
        Tuple containing feature and target values for testing
    config: dict
        Dictionary of network configuration parameters
    n_features: int
        Number of features (inputs) for configuring input layer of network
    n_targets: int
        Number of targets (outputs) for configuring output layer of network
    plots: bool
        Switch for generating diagnostic plots after each network training

    Returns
    -------
    net: theanets.Classifier object
        Neural network object
    accuracy: float
        Accuracy value of the network configuration from the validation dataset
    monitors: dict
        Dictionary of "monitor" objects produced during network training
        Contains two labels 'train' and 'valid' with the following attributes:
            - 'loss': percentage from loss function (default: cross-entropy)
            - 'err': percentage of error (default: )
            - 'acc': percentage of accuracy (defualt: true classifications)
    '''
    from collections import OrderedDict
    import theanets

    # Build neural net with defined configuration
    hidden_layers = [config['hidden_nodes'],]*config['hidden_layers']
    net = theanets.Classifier([n_features,] + hidden_layers + [n_targets,])

    # Uses 'mse' as loss function # TODO cross-entropy?
    #net = theanets.Regressor(layers=[n_features, config['hidden_nodes'], n_targets])

    # mini-batchs
    # http://sebastianruder.com/optimizing-gradient-descent/index.html#minibatchgradientdescent
    # https://github.com/lmjohns3/theanets/blob/master/scripts/theanets-char-rnn


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


    def plot_monitors(attrs, monitors_train, monitors_valid):
        import matplotlib.pyplot as plt
        import seaborn

        seaborn.set_style('whitegrid')

        labels = {'loss':'Loss', 'err':'Error', 'acc':'Accuracy'}

        fig, axes = plt.subplots(1, len(attrs), sharex=True)
        legend_on = True
        for ax, attr in zip(axes, attrs):
            ax.yaxis.label.set_text(labels[attr])
            ax.xaxis.label.set_text('Epic')
            ax.plot(monitors['train'][attr], label='Training')
            ax.plot(monitors['valid'][attr], label='Validation')
            if legend_on:
                ax.legend(loc='upper left')
                legend_on = False
        plt.show()

        # Keep seaborn from messing up confusion matrix plots
        seaborn.reset_orig()

        return None

    # SGD converges to minima/maxima faster with momentum
    # NAG, ADADELTA, RMSProp have equivalents of parameter specific momentum
    if config['algorithm'] is 'sgd':
        config['momentum'] = 0.9

    # Create dictionary for storing monitor lists
    attrs = ['loss', 'err', 'acc']
    monitors = OrderedDict()
    for mtype in ['train', 'valid']:
        monitors[mtype] = dict()
        for attr in attrs:
            monitors[mtype][attr] = list()

    print('')
    print('Train samples:       {:8d}'.format(len(train[0])))
    print('Valididation samples:{:8d}'.format(len(valid[0])))
    print('Hidden layers:       {:8d}'.format(config['hidden_layers']))
    print('Hidden nodes/layer:  {:8d}'.format(config['hidden_nodes']))
    print('')

    kwargs = {'train':train,
              'valid':valid,
              'algo':config['algorithm'],
              #'patience':config['patience'],
              #'min_improvement':config['min_improvement'],
              #'validate_every':config['validate_every'],
              #'batch_size':batch_size,
              #'train_batches':train_batches,
              #'valid_batches':valid_batches,
              'learning_rate':config['learning_rate'],
              'momentum':config['momentum'],
              'hidden_l1':config['hidden_l1'],
              'weight_l2':config['weight_l2']}

    # Run with monitors if `plots` flag set to true
    if plots == True:
        for t_monitors, v_monitors in net.itertrain(**kwargs):
            for key in attrs:
                monitors['train'][key].append(t_monitors[key])
                monitors['valid'][key].append(v_monitors[key])

        plot_monitors(attrs, monitors['train'], monitors['valid'])

    # Run with `train` wrapper of `itertrain`
    else:
        net.train(**kwargs)

    # Classify features against label/target value to get accuracy
    # where `valid` is a tuple with validation (features, label)
    accuracy = net.score(valid[0], valid[1])

    return net, accuracy, monitors


def get_best(results, key):
    '''Return results column 'key''s value from model with best accuracy'''
    best_idx = results['accuracy'].idxmax()
    return results[key][best_idx]


def tune_net(train, valid, test, targets, configs, n_features, n_targets, plots):
    '''Train nets with varying configurations and `validation` set

    The determined best configuration is then used to find the resulting
    accuracy with the `test` dataset

    Args
    ----
    train: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training
    valid: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training validation
    test: tuple (ndarray, ndarray)
        Tuple containing feature and target values for testing
    targets: ndarray
        List of unique targets (bins) generated during data splitting
    configs: dict
        Dictionary of all permutation of network configuration parameters
        defined in `cfg_ann.yaml` file
    n_features: int
        Number of features (inputs) for configuring input layer of network
    n_targets: int
        Number of targets (outputs) for configuring output layer of network
    plots: bool
        Switch for generating diagnostic plots after each network training

    Returns
    -------
    results_tune: pandas.DataFrame (dtype=object)
        Dataframe with columns for each different configuration:
            * parameter configuration
            * network object
            * accuracy value from validation set
            * training time.
    test_accuracy: float
        Accuracy value of best configuration from test dataset
    cms: dict
        Dictionary of confusion matrices for labels 'train' & 'valid'. These
        matrices are generated from the most optimal tuning network
        configuration.
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

        net, accuracy, monitors = create_algorithm(train, valid, configs[i],
                                                   n_features, n_targets, plots)

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
    print('Tuning test accuracy: {}'.format(test_accuracy))

    # Print confusion matrices for train and test
    cms = get_confusion_matrices(best_net, train, test, targets)

    return results_tune, test_accuracy, cms


def truncate_data(data, frac):
    '''Reduce data rows to `frac` of original

    Args
    ----
    data: Tuple containing numpy array of feature data and labels
    frac: percetange of original data to return

    Returns
    -------
    subset_frac: pandas.DataFrame
        Slice of original dataframe with len(data)*n rows
    '''

    n = len(data[0])
    subset_frac = (data[0][:round(n*frac)], data[1][:round(n*frac)])

    return subset_frac


def test_dataset_size(best_config, train, valid, test, targets, n_features, n_targets,
        subset_fractions):
    '''Train nets with best configuration and varying dataset sizes

    Args
    ----
    best_config: dict
        Dictionary of configuration parameters for best performing network
    train: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training
    valid: tuple (ndarray, ndarray)
        Tuple containing feature and target values for training validation
    test: tuple (ndarray, ndarray)
        Tuple containing feature and target values for testing
    targets: ndarray
        List of unique targets (bins) generated during data splitting
    n_features: int
        Number of features (inputs) for configuring input layer of network
    n_targets: int
        Number of targets (outputs) for configuring output layer of network

    Returns
    -------
    results_dataset: pandas.DataFrame (dtype=object)
        Dataframe with columns for each different configuration:
            * parameter configuration
            * network object
            * accuracy value from validation set
            * fraction of original dataset
            * training time.
    test_accuracy: float
        Accuracy value of best configuration from test dataset
    cms: dict
        Dictionary of confusion matrices for labels 'train' & 'valid'. These
        matrices are generated from the most optimal dataset size network
        configuration.
    '''
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

        net, accuracy, monitors = create_algorithm(train, valid, best_config,
                                                   n_features, n_targets)
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

    # Print confusion matrices for train and test
    cms = get_confusion_matrices(best_net, train, test, targets)

    return results_dataset, test_accuracy, cms


def run(file_cfg_paths, path_cfg_ann, debug=False, plots=False):
    '''
    Compile subglide data, tune network architecture and test dataset size

    Args
    ----
    file_cfg_paths: str
        Full path to `cfg_paths.yaml` file
    path_cfg_ann: str
        Full path to `cfg_ann.yaml` file
    debug: bool
        Swith for running single network configuration
    plots: bool
        Switch for generating diagnostic plots after each network training

    Returns
    -------
    cfg: dict
        Dictionary of network configuration parameters used
    data: tuple
        Tuple collecting training, validation, and test sets. Also includes bin
        deliniation values
    results: tuple
        Tuple collecting results dataframes and confusion matrices

    Note
    ----
    The validation set is split into `validation` and `test` sets, the
    first used for initial comparisons of various net configuration
    accuracies and the second for a clean test set to get an true accuracy,
    as reusing the `validation` set can cause the routine to overfit to the
    validation set.
    '''

    # TODO add starttime, finishtime, git/versions

    from collections import OrderedDict
    import climate
    import datetime
    import numpy
    import os
    import pickle
    import theano

    import utils_data
    from rjdtools import yaml_tools

    # Environment settings - logging, Theano, load configuration, set paths
    #---------------------------------------------------------------------------
    climate.enable_default_logging()
    theano.config.compute_test_value = 'ignore'

    # Configuration settings
    cfg = yaml_tools.read_yaml(path_cfg_ann)
    if debug is True:
        for key in cfg['net_tuning'].keys():
            cfg['net_tuning'][key] = [cfg['net_tuning'][key][0],]

    # Define output directory and create if does not exist
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg['output']['results_path'] = 'theanets_{}'.format(now)

    # Define paths
    paths = yaml_tools.read_yaml(file_cfg_paths)

    path_root       = paths['root']
    path_acc        = paths['acc']
    glide_path      = paths['glide']
    path_ann        = paths['ann']
    path_bc         = paths['bodycondition']
    fname_bc        = 'coexist_experiments.p'
    fname_sgls      = 'data_sgls.p'
    fname_mask_sgls = 'mask_sgls_filt.p'
    fname_ann_sgls  = 'sgls_all.p'
    fname_ind_train = 'ind_train.p'
    fname_ind_valid = 'ind_valid.p'
    fname_ind_test  = 'ind_test.p'

    # Print input data configuration
    print_dict_values(cfg['data'])


    # Compile, split, and normalize data
    #---------------------------------------------------------------------------
    sgl_cols = cfg['data']['sgl_cols'] + cfg['net_all']['features']

    # Compile output from glides into single input dataframe
    _, sgls, _ = utils_data.create_ann_inputs(path_root,
                                              path_acc,
                                              glide_path,
                                              path_ann,
                                              path_bc,
                                              fname_bc,
                                              fname_sgls,
                                              fname_mask_sgls,
                                              sgl_cols,
                                              manual_selection=True)

    # Drop rows missing data
    sgls = sgls.dropna()

    print('\nSplit and normalize input/output data')
    features   = cfg['net_all']['features']
    target     = cfg['net_all']['target']
    n_targets  = cfg['net_all']['n_targets']
    valid_frac = cfg['net_all']['valid_frac']

    # Normalize input (features) and output (target)
    sgls, bins = normalize_data(sgls, features, target, n_targets)
    bins_list = [float(b) for b in bins]

    # Get indices of train, validation and test datasets
    ind_train, ind_valid, ind_test = get_split_indexes(sgls, valid_frac)

    # Split dataframes into train, validation and test  (features, targets) tuples
    train, valid, test = create_datasets(sgls, ind_train, ind_valid, ind_test,
                                         features, target)

    # Save information on input data to config
    cfg['net_all']['n_train'] = len(train[0])
    cfg['net_all']['n_valid'] = len(valid[0])
    cfg['net_all']['n_test'] = len(test[0])
    cfg['net_all']['targets'] = bins_list


    # Tuning - find optimal network architecture
    #---------------------------------------------------------------------------
    print('\nTune netork configuration')

    # Get all dict of all configuration permutations of params in `tune_params`
    configs = get_configs(cfg['net_tuning'])

    # Cycle through configurations storing configuration, net in `results_tune`
    n_features = len(cfg['net_all']['features'])
    n_targets = cfg['net_all']['n_targets']

    print('\nNumber of features: {}'.format(n_features))
    print('Number of targets: {}\n'.format(n_targets))

    results_tune, tune_accuracy, tune_cms = tune_net(train,
                                                     valid,
                                                     test,
                                                     bins,
                                                     configs,
                                                     n_features,
                                                     n_targets,
                                                     plots)

    # Get neural net configuration with best accuracy
    best_config = get_best(results_tune, 'config')


    # Test effect of dataset size
    #---------------------------------------------------------------------------
    print('\nRun percentage of datasize tests')

    # Get randomly sorted and subsetted datasets to test effect of dataset_size
    # i.e. - a dataset with the first `subset_fraction` of samples.
    subset_fractions = cfg['net_dataset_size']['subset_fractions']
    results_dataset, data_accuracy, data_cms = test_dataset_size(best_config,
                                                                 train,
                                                                 valid,
                                                                 test,
                                                                 bins,
                                                                 n_features,
                                                                 n_targets,
                                                                 subset_fractions)

    print('\nTest data accuracy (Configuration tuning): {}'.format(tune_accuracy))
    print('Test data accuracy (Datasize test):        {}'.format(data_accuracy))


    # Save results and configuration to output directory
    #---------------------------------------------------------------------------
    # Create output directory if it does not exist
    out_path = os.path.join(path_root, path_ann, cfg['output']['results_path'])
    os.makedirs(out_path, exist_ok=True)

    # Save config as a `*.yaml` file to the output directory
    yaml_tools.write_yaml(cfg, os.path.join(out_path, path_cfg_ann))

    # Save output data to analysis output directory
    results_tune.to_pickle(os.path.join(out_path, cfg['output']['tune_fname']))
    results_dataset.to_pickle(os.path.join(out_path, cfg['output']['dataset_size_fname']))

    # Save the generated input data to the analysis output directory
    sgls.to_pickle(os.path.join(out_path, fname_ann_sgls))

    # Save indices for creating the same train, validation, test datasets
    pickle.dump(train, open(os.path.join(out_path, fname_ind_train), 'wb'))
    pickle.dump(valid, open(os.path.join(out_path, fname_ind_valid), 'wb'))
    pickle.dump(test,  open(os.path.join(out_path, fname_ind_test), 'wb'))

    return cfg, (train, valid, test, bins), (results_tune, results_dataset,
                                             tune_cms, data_cms)


def print_results_accuracy(results_dataset, test):
    '''Print accuracies of all nets in results on test dataset'''

    for i, net in enumerate(results_dataset['net']):
        accuracy = net.score(test[0], test[1])
        print('[{}] accuracy: {}'.format(i, accuracy))

    return None

if __name__ == '__main__':
    #cfg, results_tune, results_dataset = run()

    debug = False
    plots = False

    file_cfg_paths = './cfg_paths.yaml'
    path_cfg_ann   = './cfg_ann.yaml'

    cfg, data, results = run(file_cfg_paths, path_cfg_ann, debug=debug,
                             plots=plots)

    train = data[0]
    valid = data[1]
    test  = data[2]
    bins  = data[3]

    results_tune    = results[0]
    results_dataset = results[1]
    tune_cms        = results[2]
    data_cms        = results[3]
