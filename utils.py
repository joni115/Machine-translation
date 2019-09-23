import glob
import os


from gensim.utils import simple_preprocess
from zipfile import ZipFile

def pre_proccess_line(line):
    return simple_preprocess(line.strip())

def read_zip(zipfile='text/cookbook_text.zip'):
    zipfile = ZipFile(zipfile, 'r')
    for filename in zipfile.namelist():
        text = zipfile.read(filename)
        yield pre_proccess_line(text)

def read_file(filename='text'):
    with open(filename, 'r') as f:
        text = pre_proccess_line(f.read())
    return text

def read_files_in_directory(directory='text/shakespeare'):
    files = glob.glob(directory + '/*.txt')
    for filename in files:
        yield read_file(filename)

def load_corpus(corpus=''):
    global switcher

    if corpus in switcher.keys():
        return switcher[corpus]
    else:
        raise NameError

def add_corpus(corpus, path):
    global switcher

    if not os.path.isfile(path):
        print('{} is not a valid file'.format(path))
        return

    switcher[corpus] = read_file(path)
    print('corpus added.')

switcher = {
    'cookbook': read_files_in_directory('text/cookbook_text'),
    'shakespeare': read_files_in_directory('text/shakespeare'),
    'wikipedia': read_files_in_directory('text/wikipedia')
}