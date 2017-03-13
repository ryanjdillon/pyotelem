import datetime
import os
import sys

class Log(object):
    '''Write plain text log files

    http://stackoverflow.com/a/5916874/943773
    '''
    def __init__(self, out_path, suffix, term_on=True, write_on=True):
        fmt = 'log_%Y-%m-%d_%H%M%S_{}.txt'.format(suffix)
        file_name = datetime.datetime.now().strftime(fmt)
        self.path = os.path.join(out_path, file_name)
        self.term_on = term_on
        self.write_on = write_on


    def new_entry(self, msg):
        with open(self.path, 'a') as f:
            log_line = '{}, {}\n'.format(datetime.datetime.now(), msg)
            if self.term_on:
                print(log_line)
            if self.write_on:
                f.write(log_line)


def log_times(path):
    '''Get sorted list of log items which took most time'''
    import numpy

    dates = list()
    for l in logtxt:
         dates.append(datetime.datetime.strptime(l.split(',')[0],
                                                 '%Y-%m-%d_%H%M%S'))
    dates = numpy.asarray(dates)
    time_diff = numpy.diff(dates)

    return dates
