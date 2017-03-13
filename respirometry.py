
def read_header(fileio):
    '''Read information rows from respirometry files

    Example
    -------
    Interval=	0.010000 s
    ExcelDateTime=	4.2477442185022897e+004	04/17/16 10:36:44.785978
    TimeFormat=	StartOfBlock
    DateFormat=
    ChannelTitle=	O2	Differential pressure
    Range=	10.000 V	10.000 V
    UnitName=	%O2	*
    TopValue=	156.76	*
    BottomValue=	-291.24	*
    '''
    from collections import OrderedDict
    import datetime

    val = lambda fileio: fileio.readline().split('=')[1].strip()

    header = OrderedDict()
    header['interval'] = float(val(fileio)[:-2])

    fmt = '%m/%d/%y %H:%M:%S.%f'
    dt_str = ' '.join(val(fileio).split()[1:])
    header['datetime']      = datetime.datetime.strptime(dt_str, fmt)

    header['time_format']   = val(fileio)
    header['date_format']   = val(fileio)

    header['channel_title'] = val(fileio).replace('\t',' ')

    val_range = val(fileio).split('\t')
    header['range_start']   = val_range[0]
    header['range_stop']    = val_range[1]

    header['unit']          = val(fileio).split('\t')[0]
    header['top_value']     = val(fileio).split('\t')[0]
    header['bottom_value']  = val(fileio).split('\t')[0]

    return fileio, header


def create_datetimes(headers, n_rows):
    '''Generate datetime index from time, interval, n_values in data'''
    import datetime
    import numpy
    import copy

    datetimes = numpy.zeros(n_rows, dtype=object)

    # Process each data block with associated header information
    for i in range(len(headers)):
        delta  = datetime.timedelta(seconds=float(headers[i]['interval']))
        start_idx = headers[i]['file_start_idx'] - 9*i
        if i < len(headers)-1:
            end_idx = headers[i+1]['file_start_idx'] - 9*(i+1)
        else:
            end_idx = n_rows

        n_datetimes = end_idx - start_idx

        datetimes[start_idx:end_idx] = headers[i]['datetime']
        datetimes[start_idx:end_idx] += numpy.arange(n_datetimes)* delta

    return datetimes


def read_respirometry(file_path):
    '''Read labview text output data file'''
    import datetime
    import pandas
    import numpy

    import utils

    # Create oversized numpy array for data
    n_lines = utils.get_n_lines(file_path)

    data = numpy.zeros((n_lines,4), dtype=object)#n_lines, dtype=dtypes)
    data[:] = numpy.nan

    f = open(file_path, 'r')

    headers = list()

    data_idx = 0
    line_idx = 0
    while True:
        # Save current file position to return to if reading header
        f_pos = f.tell()

        line=f.readline()

        # If EOF reached, break while loop
        if not line: break

        # If a header block is reached, read header info, save in list
        if '=' in line:
            f.seek(f_pos)
            f, header = read_header(f)
            header['file_start_idx'] = line_idx
            headers.append(header)

        # Else parse out data values and save in data array
        else:
            for j, val in enumerate(line.split('\t')):
                # If 4th element exists, treat as string note
                if j == 3:
                    data[data_idx,j] = val[3:].strip()
                else:
                    data[data_idx,j] = float(val)
            data_idx += 1
        line_idx += 1

    f.close()

    val_mask = ~numpy.isnan(data[:,0].astype(float))
    # filter excess rows without data from data array
    data = data[val_mask, :]

    datetimes = create_datetimes(headers, len(data))

    data = numpy.vstack([datetimes, data[:,1:].T]).T

    # create pandas DataFrame with datetime index
    resp_df = pandas.DataFrame(data, columns=['datetimes','a','b','note'])

    return resp_df

if __name__ == '__main__':
    import os

    from rjdtools import yaml_tools

    paths = yaml_tools.read_yaml('./cfg_paths.yaml')
    resp_path = os.path.join(paths['root'], paths['respirometry'])

    for d in os.listdir(resp_path):
        if d.endswith('.txt'):
            print('Processing file: {}'.format(d))
            file_path = os.path.join(resp_path, d)
            file_name = os.path.splitext(d)[0]
            out_path = os.path.join(resp_path, file_name+'.p')
            df = read_respirometry(file_path)
            df.to_pickle(out_path)
