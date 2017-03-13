'''
DATE (UTC)	DEPTH (m)	TEMP (C)	ODO_DELTA (m)	MAGX (uT)	MAGY (uT)	MAGZ (uT)	MAG_TEMP (C)	DRYNESS
[TIME 2016/02/19 13:04:34.010] 0
Dat[1][0]=1455887074.01 2016/02/19 13:04:34.010	20.0	4.95	0.00	-55.5	22.4	41.2-18	51	
...
read_time: 2016/02/19 13:05:28 + 1/100 = 1455887128.010000 (74 02 13 0d 05 1c
01)
known_t[1]=1455887128.01 [2016/02/19 13:05:28.010] samples=108
[TIME 2016/02/19 13:05:28.010] 0
Dat[2][0]=1455887128.01 2016/02/19 13:05:28.010 20.5    4.85    0.00    -50.1
-1.3    -44.1   -19 45  

'''

def __create_dataframe(a, col_names):
    '''Convert to datetime indexed dataframe and remove NaNs'''
    import pandas

    # Copy contents to pandas
    dates = pandas.to_datetime(a[:, 0])
    df = pandas.DataFrame(data=a[:,1:], index=dates, columns=col_names[1:])

    # Drop NaN rows, typecase data columns to numerics
    df = df.dropna()
    df = df.apply(pandas.to_numeric)

    return df


def __write_dataframe(df, out_path, filename, write_pickle=True,
        write_feather=True):
    '''Write dataframe to pickle and or feather formats'''
    import feather
    import pandas

    # Write files to pickle
    pickle_path  = os.path.join(out_path, 'pandas')
    os.makedirs(pickle_path, exist_ok=True)
    if write_pickle:
        pickle_name  = '{}.p'.format(filename)
        print('Writing datafile: {}'.format(pickle_name))
        df.to_pickle(os.path.join(pickle_path, pickle_name))

    # Write files to feather
    feather_path = os.path.join(out_path, 'feather')
    os.makedirs(feather_path, exist_ok=True)
    if write_feather:
        feather_name = '{}.f'.format(filename)
        print('Writing datafile: {}'.format(feather_name))
        feather.write_dataframe(df, os.path.join(feather_path, feather_name))

    return None


def __split_dates_write(df, out_path, suffix, write_pickle=False, write_feather=False):
    '''Split dataframe by unique date and write to pickle or feather files'''
    import numpy
    import pandas

    # Split files, save as ind pickles
    df_dict = dict()
    dates   = df.index.map(pandas.Timestamp.date)

    for dt in numpy.unique(dates):
        mask   = dates == dt
        dt_str = dt.strftime('%Y-%m-%d')
        filename = '{}_{}'.format(dt_str, suffix)

        __write_dataframe(df[mask], out_path, filename,
                          write_pickle=write_pickle,
                          write_feather=write_feather)

        # Store in dictionary to return
        df_dict[dt_str] = df[mask]

    return df_dict


def read_acc(acc_txt, out_path, write_pickle=True, write_feather=True):
    '''Read martin accelerometer'''
    import feather
    import pandas
    import numpy
    import os

    from bodycondition import utils

    # Get a string date in format `20160131` from pandas datetime
    get_date = lambda x: x.date().strftime('%Y-%m-%d')

    # Function for parsing data line
    parse_line = lambda x: x.strip().lower().split('\t')

    # Open acc file for reading
    print('Reading data file to memory: {}'.format(acc_txt))
    f = open(acc_txt, 'r')

    # Skip header row
    f.readline()

    # Column names and date format
    names = ['date', 'x', 'y', 'z', 'count', 'c1', 'c2', 'c3', 'c4']
    fmt = '%Y/%m/%d %H:%M:%S.%f'

    print('Reading and parsing accelerometer data')

    # Get number of lines, create temp numpy array for storing values
    n_lines = utils.get_n_lines(acc_txt)
    a = numpy.zeros((n_lines, len(names)), dtype=object)
    a[:] = numpy.nan

    # Process all lines
    i = 0
    i_line = 0
    acc_dict = dict()
    for line in f:
    #for j in range(100000):
    #    line = f.readline()

        # Only process if not an intermittent  header row
        if not line.startswith('[TIME'):
            vals = parse_line(line)
            vals[0] = pandas.to_datetime(vals[0], format=fmt)

            dt_str = get_date(vals[0])
            suffix = 'acc'
            filename = '{}_{}'.format(dt_str, suffix)

            # Check if Date has changed, save dataframe to dict, reset `a`
            if i != 0:
                if vals[0].date() != a[i-1,0].date():

                    print('Adding dataframe to dict: {}'.format(dt_str))

                    # Convert array to dataframe without NaNs and write to out_path
                    acc_dict[dt_str] = __create_dataframe(a, names)
                    __write_dataframe(acc_dict[dt_str],
                                      out_path,
                                      filename,
                                      write_pickle=write_pickle,
                                      write_feather=write_feather)

                    # Clear array of values and reset index counter
                    a[:] = numpy.nan
                    i = 0

            # Write current values to array
            try:
                a[i,:] = vals
                i += 1
            except:
                raise SystemError('Error occured on data file line '
                                 '{}'.format(i_line))
        i_line +=1

    # Catch last date processed, or if only a single date in file
    if get_date(vals[0]) not in acc_dict.keys():
        acc_dict[dt_str] = __create_dataframe(a, names)
        __write_dataframe(acc_dict[dt_str],
                          out_path,
                          filename,
                          write_pickle=write_pickle,
                          write_feather=write_feather)

    # Close acc file
    f.close()

    return acc_dict


def read_tdr(tdr_txt, out_path, write_pickle=True, write_feather=True):
    '''Read martin tdr'''
    import datetime
    import feather
    import numpy
    import os
    import pandas

    from bodycondition import utils

    # Get number of lines of file for approx size of dataframe
    n_lines = utils.get_n_lines(tdr_txt)

    # Read in header row
    f = open(tdr_txt, 'r')
    head = f.readline().split()
    names = [name.lower() for name in head[::2]]
    units = head[1::2]

    # Create empty dateframe for filling
    a = numpy.zeros((n_lines, len(names)), dtype=object)
    a[:] = numpy.nan

    # skip pointless date row
    f.readline()

    # String format of combined 'date' & 'time' columns
    fmt = '%Y/%m/%d %H:%M:%S.%f'

    print('Reading and parsing accelerometer data')

    # Process each line of file after first header rows
    # Set counter for actual rows of data (not headers)
    i = 0
    for row in f:
        # Process row if starts wit 'Date', else skip
        if row.startswith('Dat'):
            # Parse row and create datetime for current row, list of values
            row = row.split()
            dt_str = ' '.join(row[1:3])

            vals = row[3:]
            vals.insert(0, dt_str)
            a[i, :] = vals

            i += 1

    # Convert to datetime indexed dataframe and remove NaNs
    tdr = __create_dataframe(a, names)

    tdr_dict = __split_dates_write(tdr, out_path, 'tdr',
                                  write_pickle=write_pickle,
                                  write_feather=write_feather)
    return tdr_dict


def __merge_dataframes(df1, df2):
    '''Merge dataframes at different sampling rates, interpolate and fill'''

    merged = df1.join(df2, how='outer')
    merged = merged.interpolate('time')
    merged = merged.fillna(method='backfill')

    return merged



if __name__ == '__main__':
    import os
    import numpy

    root_path = ('/home/ryan/Desktop/edu/01_PhD/projects/smartmove/data_coexist/'
                 'data_acc_not-glide-analyzed/SMRU_12707_acc')

    out_path = root_path

    acc_txt = os.path.join(root_path, 'martin12707_accel.txt')
    tdr_txt = os.path.join(root_path, 'corrected_tdr.txt')

    acc_dict = read_acc(acc_txt, out_path, write_pickle=True,
                                           write_feather=False)

    tdr_dict = read_tdr(tdr_txt, out_path, write_pickle=True,
                                           write_feather=False)

    # Merge dataframes and write to file
    for key in numpy.unique(list(acc_dict.keys())):
        if key in tdr_dict:
            merged = __merge_dataframes(acc_dict[key], tdr_dict[key])
            merge_fname = os.path.join(out_path, '{}_merged.p'.format(key))
            merged.to_pickle(merge_fname)

    tdr_path = os.path.join(out_path, 'tdr')
    acc_path = os.path.join(out_path, 'acc')
    ext_list = lambda x: [d for d in os.listdir(x) if d.endswith('.p')]
    accs = ext_list(acc_path)
    tdrs = ext_list(tdr_path)
