import utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.decomposition import PCA
from gensim.models import Word2Vec


w2v = ''

class Word2vec:
    def __init__(self, vector_size=300, min_count=1, window=10, path=''):
        if path:
            self.model = Word2Vec.load(path)
        else:
            self.model = Word2Vec(size=vector_size, min_count=min_count, window=window)

    def train(self, corpus, epochs=5):
        self.model.build_vocab(corpus)
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=5)

    def save_model(self, path):
        self.model.save(path)

    def get_vocabulary(self):
        return self.model.wv.vocab.keys()

    def graphic_words(self, words=None, sample=0):
        if not words:
            if sample > 0:
                words = np.random.choice(list(self.get_vocabulary()), sample)
            else:
                words = [ word for word in model.vocab ]
        
        word_vectors = np.array([self.model[w] for w in words])

        twodim = PCA().fit_transform(word_vectors)[:,:2]
        plt.figure(figsize=(6,6))
        plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
        for word, (x,y) in zip(words, twodim):
            print(word, (x,y))
            plt.text(x, y, word)
        plt.show()
        return words


def save_model(path):
    global w2v

    if not w2v:
        print('please train or load a model')
        return
    w2v.save_model(path)


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


def get_vocabulary(top=10):
    global w2v

    if not w2v:
        please('please train or load a model')
        return

    return pd.DataFrame(w2v.get_vocabulary(), columns=['word']).head(top)


def cosine_similarity_between_words(word1, word2):
    global w2v

    if not w2v:
        please('please train or load a model')
        return
    try:
        cosine_similarity = w2v.model.wv.similarity(word1, word2)
    except KeyError as e:
        print(e)
        return
    return cosine_similarity


def graphic_words(words=None, sample=0):
    global w2v

    if not w2v:
        please('please train or load a model')
        return

    w2v.graphic_words(words, sample)