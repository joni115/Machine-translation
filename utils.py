import glob

from gensim.utils import simple_preprocess
from zipfile import ZipFile


def pre_proccess_line(line):
    return simple_preprocess(line.strip())

def read_zip(zipfile='cookbook_text.zip'):
    zipfile = ZipFile(zipfile, 'r')
    for filename in zipfile.namelist():
        text = zipfile.read(filename)
        yield pre_proccess_line(text)

def read_files_in_directory(directory='shakespeare'):
    files = glob.glob(directory + '/*.txt')
    for filename in files:
        with open(filename, 'r') as f:
            text = f.read()
        yield pre_proccess_line(text)

def load_corpus(corpus=''):
    switcher = {
        'cookbook': read_zip(),
        'shakespeare': read_files_in_directory(),
    }
    if corpus in switcher.keys():
        return switcher[corpus]
    else:
        raise NameError