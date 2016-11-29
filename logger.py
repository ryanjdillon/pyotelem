import datetime
import os

class Log(object):
    '''Write plain text log files'''
    def __init__(self, out_path, printon=True, writeon=True):
        file_name = datetime.datetime.now().strftime('log_%Y-%m-%d_%H%M%S.txt')
        self.path = os.path.join(out_path, file_name)
        self.printon = printon
        self.writeon = writeon

    def new_entry(self, msg):
        with open(self.path, 'a') as f:
            log_line = '{}, {}\n'.format(datetime.datetime.now(), msg)
            if self.printon:
                print(log_line)
            if self.writeon:
                f.write(log_line)
