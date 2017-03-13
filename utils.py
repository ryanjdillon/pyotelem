
def cat_keyvalues(d, ignore):
    '''Concatenate dictionary key, value pairs to a single string'''

    items = list(d.items())
    s = ''
    for i in range(len(items)):
        key, value = items[i]
        if key not in set(ignore):
            s += '{}_{}__'.format(key, value)

    return s[:-2]


def parse_subdir(path):
    '''Parse parameters in directory names to pandas dataframe

    Parameters are separated by double `_` and values by single. Names that
    include an `_` are joined back together after they are split
    '''
    import os
    import numpy
    import pandas

    dir_list = numpy.asarray(os.listdir(path), dtype=object)

    # Search root directory for directories to parse
    for i in range(len(dir_list)):
        if os.path.isdir(os.path.join(path,dir_list[i])):
            name = dir_list[i]
            # Split parameters in name
            dir_list[i] = dir_list[i].split('__')
            for j in range(len(dir_list[i])):
                param = dir_list[i][j].split('_')
                # Join names with `_` back together, make key/value tuple
                key = '_'.join(param[:-1])
                value = param[-1]
                if value == 'None':
                    value = numpy.nan
                param = (key, float(value))
                dir_list[i][j] = param
            # Convert list of tuples to dictionary
            dir_list[i] = dict(dir_list[i])
            # Add directory name to dict for later retrieval
            dir_list[i]['name'] = name
        else:
            dir_list[i] = ''

    # Remove entries that are files
    dir_list = dir_list[~(dir_list == '')]

    # Convert list of dictionaries to dictionary of lists
    keys = dir_list[0].keys()
    params = dict()
    for i in range(len(dir_list)):
        for key in dir_list[i]:
            if key not in params:
                params[key] = numpy.zeros(len(dir_list), object)
            params[key][i] = dir_list[i][key]

    return pandas.DataFrame(params)

def get_versions(module_name):
    '''Return versions for repository and packages in requirements file

    Args
    ----
    module_name: str
        Name of module calling this routine, stored with local git hash

    Returns
    -------
    versions: OrderedDict
        Dictionary of module name and dependencies with versions
    '''
    from collections import OrderedDict
    import importlib
    import os

    versions = OrderedDict()

    module = importlib.util.find_spec(module_name)

    # Add git hash for pylleo to dict
    versions[module_name] = get_githash('long')

    # Get path to pylleo requirements file
    module_path  = os.path.split(module.origin)[0]
    requirements = os.path.join(module_path, 'requirements.txt')

    # Add packages and versions to dictionary
    with open(requirements) as f:
        for l in f.readlines():
            package, version = l.strip().split('==')
            versions[package] = version

    return versions


def get_githash(hash_type):
    '''Add git commit for reference to code that produced data

    Args
    ----
    hash_type: str
        keyword determining length of has. 'long' gives full hash, 'short'
        gives 6 character hash

    Returns
    -------
    git_hash: str
        Git hash as a 6 or 40 char string depending on keywork `hash_type`
    '''
    import subprocess

    cmd = dict()
    cmd['long']  = ['git', 'rev-parse', 'HEAD']
    cmd['short'] = ['git', 'rev-parse', '--short', 'HEAD']

    return subprocess.check_output(cmd[hash_type]).decode('ascii').strip()


def nearest(items, pivot):
    '''Find nearest value in array, including datetimes'''
    return min(items, key=lambda x: abs(x - pivot))


def add_col(df, col_name, values):
    '''Add column `col_name` dataframe with values

    df = add_col(df, col_name, numpy.full(len(df), numpy.nan))
    '''
    return df.assign(**{col_name:values})


def get_n_lines(file_path):
    '''Get number of lines by calling bash command wc'''
    import os
    import subprocess

    cmd = 'wc -l {0}'.format(file_path)
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    n_lines = int((output).readlines()[0].split()[0])

    return n_lines


def mask_from_noncontiguous_indices(n, start_ind, stop_ind):
    '''Create boolean mask from start stop indices of noncontiguous regions

    Args
    ----
    n: int
        length of boolean array to fill
    start_ind: numpy.ndarray
        start index positions of non-contiguous regions
    stop_ind: numpy.ndarray
        stop index positions of non-contiguous regions

    Returns
    -------
    mask: numpy.ndarray, shape (n,), dtype boolean
        boolean mask array
    '''
    import numpy

    mask = numpy.zeros(n, dtype=bool)

    for i in range(len(start_ind)):
        mask[start_ind[i]:stop_ind[i]] = True

    return mask


def contiguous_regions(condition):
    '''Finds contiguous True regions of the boolean array 'condition'.

    Args
    ----
    condition: numpy.ndarray, dtype bool
        boolean mask array, but can pass the condition itself (e.g. x > 5)

    Returns
    -------
    start_ind: numpy.ndarray, dtype int
        array with the start indices for each contiguous region
    stop_ind: numpy.ndarray, dtype int
        array with the stop indices for each contiguous region

    http://stackoverflow.com/a/4495197/943773
    '''
    import numpy

    if condition.ndim > 1:
        raise IndexError('contiguous_regions(): condition must be 1-D')

    # Find the indicies of changes in 'condition'
    idx = numpy.diff(condition).nonzero()[0]

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = numpy.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = numpy.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)

    # We need to start things after the change in 'condition'. Therefore,
    # we'll shift the index by 1 to the right.
    start_ind = idx[:,0] + 1
    # keep the stop ending just before the change in condition
    stop_ind  = idx[:,1]

    # remove reversed or same start/stop index
    good_vals = (stop_ind-start_ind) > 0
    start_ind = start_ind[good_vals]
    stop_ind = stop_ind[good_vals]

    return start_ind, stop_ind


def rm_regions(a, b, a_start_ind, a_stop_ind):
    '''Remove additional contiguous regions in `a` that occur before a
    complimentary region in `b` has occured'''
    import numpy

    for i in range(len(a_stop_ind)):
        next_a_start = numpy.argmax(a[a_stop_ind[i]:])
        next_b_start = numpy.argmax(b[a_stop_ind[i]:])
        if  next_b_start > next_a_start:
            a[a_start_ind[i]:a_stop_ind[i]] = False

        return a


def recursive_input(input_label, type_class):
    '''General user input function

    Args
    ----
    type_class (type): name of python type (e.g. float, no parentheses)

    Returns
    -------
    output: value entered by user converted to type `type_class`

    Note
    ----
    Use `ctrl-c` to exit input cycling
    '''
    import sys

    msg = 'Enter {} {}: '.format(input_label, type_class)

    # Catch `Ctrl-c` keyboard interupts
    try:
        output = input(msg)
        print('')
        # Type class input, else cycle input function again
        try:
            output = type_class(output)
            return output
        except:
            print('Input must be type {}\n'.format(type_class))
            return recursive_input(input_label, type_class)

    # Keyboard interrupt passed, exit recursive input
    except KeyboardInterrupt:
        return sys.exit()


def get_dir_indices(msg, dirs):
    '''Return path(s) indices of directory list from user input

    msg: str
        String with message to display before pass selection input
    dir_list: array-like
        list of paths to be displayed and selected from

    Return
    ------
    input_dir_indices: array-like
        list of index positions of user selected path from input
    '''
    import os

    # Get user input for paths to process
    usage = ('\nEnter numbers preceeding paths seperated by commas (e.g. '
             '`0,2,3`).\nTo select all paths type `all`.\nSingle directories '
             'can also be entered (e.g.  `0`)\n\n')

    # Generate paths text to display
    dirs_str = ['{:2} {:60}\n'.format(i, p) for i, p in enumerate(dirs)]
    dirs_str = ''.join(dirs_str)

    # Concatenate `msg`, usage txt, and paths list for display before input
    input_dirs   = recursive_input(''.join([msg, usage, dirs_str, '\n']), str)

    # Determine type of input
    if ',' in input_dirs:
        input_dir_indices = [int(x.strip()) for x in input_dirs.split(',')]
    elif 'all' in input_dirs:
        input_dir_indices = range(len(dirs))
    else:
        try:
            input_dir_indices = [int(input_dirs.strip()),]
        except:
            raise ValueError('Could not determine input type for input: '
                             '{}'.format(input_dirs))

    return input_dir_indices


def normalized(a, axis=-1, order=2):
    '''Return normalized vector for arbitrary axis

    http://stackoverflow.com/a/21032099/943773
    '''
    import numpy

    l2 = numpy.atleast_1d(numpy.linalg.norm(a, order, axis))
    l2[l2==0] = 1

    return a / numpy.expand_dims(l2, axis)
