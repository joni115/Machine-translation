import glob
import os


from gensim.utils import simple_preprocess
from zipfile import ZipFile

def pre_proccess_line(line):
    return simple_preprocess(line.strip())

def pre_process_file(file):
    preprocessed_file = []
    for line in file:
        preprocessed_line = pre_proccess_line(line)
        if not preprocessed_line:
            continue
        preprocessed_file.append(preprocessed_line)
    return preprocessed_file

def read_zip(zipfile='text/cookbook_text.zip'):
    zipfile = ZipFile(zipfile, 'r')
    for filename in zipfile.namelist():
        text = zipfile.read(filename)
        yield pre_proccess_line(text)

def read_file(filename='text'):
    with open(filename, 'r') as f:
        text = pre_process_file(f.readlines())
    return text

def read_files_in_directory(directory='text/shakespeare'):
    files = glob.glob(directory + '/*.txt')
    text = []
    for filename in files:
        text += read_file(filename)
    return text

def load_corpus(corpus=''):
    global switcher

    if corpus in switcher.keys():
        return switcher[corpus]()
    else:
        raise NameError

def add_corpus(corpus, path):
    global switcher

    if not os.path.isfile(path):
        print('{} is not a valid file'.format(path))
        return

    switcher[corpus] = lambda: read_file(path)
    print('corpus added.')

switcher = {
    'cookbook': lambda: read_files_in_directory('text/cookbook_text'),
    'shakespeare': lambda: read_files_in_directory('text/shakespeare'),
    'wikipedia': lambda: read_files_in_directory('text/wikipedia')
}
