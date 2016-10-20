__author__ = 'pralav'
import fnmatch
import string
import datetime
import joblib
from project_settings import *
from logger import setup_logging
import cPickle
import csv
import json
punct_exclude = set(string.punctuation)
table = string.maketrans("", "")

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Utils(object):
    def __init__(self, logging):
        self.logger = logging.getLogger(__name__)

    def create_folders(self,path):
        try:
            os.makedirs(path)
        except:
            pass
        return path

    def get_path(self, type, module):
        if type in DATA_TYPES and module in MODULES:
            path = "%s/%s/%s" % (BASE_DATA_PATH, type, module)
        else:
            path = "%s/%s/%s" % (BASE_DATA_PATH, type, module)
            raise Exception('Unknown Path: %s\nPlease verify the data type: %r and module type:%r' % (path,DATA_TYPES, MODULES))
        return path

    def load_file(self, path, is_csv=False, is_excel=False):
        self.logger.info("Loading Data from: %s"%path)
        idx = path.rfind(".") + 1
        if idx == 0:
            return None
        ext = path[idx:]
        try:
            if is_excel:
                f = open(path, 'rU')
            else:
                f = open(path, 'r')

            if 'pkl' in ext:
                data = cPickle.load(f)
            elif 'csv' in ext or is_csv:
                data = []
                for line in csv.reader(f, dialect=csv.excel_tab):
                    data.append(line)
            elif 'json' in ext:
                data = json.load(f)
            else:
                data = f.readlines()
            f.close()
            self.logger.info("Loaded Data from: %s"%path)
        except Exception, e:
            self.logger.error("Could not read data:%s" % str(e))

            data = None
        return data

    def save_file(self, data, path):
        idx = path.rfind(".") + 1
        idx2 = path.rfind("/")
        if idx == 0:
            return None
        ext = path[idx:]
        dir = path[:idx2]
        try:
            os.makedirs(dir)
        except:
            pass
        try:
            f = open(path, 'w')
            if 'pkl' in ext:
                data = cPickle.dump(data, f)
            elif 'csv' in ext:
                # data = []
                if isinstance(data, list):
                    writer = csv.writer(f)
                    for row in data:
                        writer.writerow(row)
                else:
                    f.write(data)

            elif 'json' in ext:
                data = json.dump(data, f,indent=4)
            else:
                data = f.write(data)
            f.close()
        except Exception, e:
            self.logger.error("Could not finish Writing data:%s" % str(e))
            data = None
        return data

    def get_files_from_folder(self, dirpath='/Users/pralav/Documents/deep_learning/datadump/res/res'):
        json_files = []
        for root, subdirs, files in os.walk(dirpath):

            for filename in fnmatch.filter(files, '*.json'):
                json_files.append(os.path.join(root, filename))
        return json_files

    def get_files_from_folder_pattern(self, dirpath='/Users/pralav/Documents/deep_learning/datadump/res/res',
                                      pattern="*term_doc_mat_*.pkl"):
        json_files = []
        for root, subdirs, files in os.walk(dirpath):

            for filename in fnmatch.filter(files, pattern):
                json_files.append(os.path.join(root, filename))
        return json_files

    def save_file_joblib(self, data, path):
        self.logger.info("Saving Data to path: %s" % path)
        try:
            dir=path[:path.rfind("/")]
            os.makedirs(dir)
        except:
            pass

        joblib.dump(data, path)

    def load_file_joblib(self, in_path):
        self.logger.info("Loading Data from path: %s" % in_path)
        data = joblib.load(in_path)
        self.logger.info("Loaded Data from path: %s" % in_path)
        return data


    def daterange(self,start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + datetime.timedelta(n)






if __name__ == '__main__':
    utils = Utils(setup_logging())
