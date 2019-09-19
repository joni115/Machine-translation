import utils

from gensim.models import Word2Vec

w2v = ''

class Word2vec:
    def __init__(self, vector_size=300, min_count=1, window=10, path=''):
        if not path:
            self.model = Word2Vec(size=vector_size, min_count=min_count, window=window)
        self.model = Word2Vec.load(path)

    def train(self, corpus, epochs=5):
        self.model.build_vocab(corpus)
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=5)


def set_parameters_to_train(vector_size=300, min_count=1, window=10):
    global w2v
    w2v = Word2vec(vector_size, min_count, window)

def train_model(corpus='cookbook', epochs=10):
    global w2v

    try:
        corpus = utils.load_corpus(corpus)
    except NameError:
        print('the corpus {0} is not available'.format(corpus))
        return

    if not w2v:
        w2v = Word2vec()
    w2v.train(corpus, epochs=epochs)
    return w2v.model.wv

def load_model(path):
    global w2v

    try:
        w2v = Word2vec(path=path)
    except FileNotFoundError:
        print("the file can't be founded")
        return
    return w2v.model.wv

def get_similar_words(word):
    global w2v

    if not w2v:
        print('please train or load a model')
        return

    try:
        most_similar = w2v.model.wv.similar_by_word(word)
    except KeyError as e:
        print(e)
        return

    return most_similar

def cosine_distance_between_words(word1, word2):
