import utils
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns


from sklearn.decomposition import PCA
from gensim.models import Word2Vec

random.seed = 43
w2v = ''

class Word2vec:
    def __init__(self, vector_size=300, min_count=1, window=10, path=''):
        if path:
            self.model = Word2Vec.load(path)
        else:
            self.model = Word2Vec(size=vector_size, min_count=min_count, window=window)

    def train(self, corpus, epochs=5):
        self.model.build_vocab(corpus)
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=epochs)

    def save_model(self, path):
        self.model.save(path)

    def get_vocabulary(self):
        return self.model.wv.vocab.keys()

    def graphic_words(self, words=None, sample=0):
        if not words:
            if sample > 0:
                words = np.random.choice(list(self.get_vocabulary()), sample)
            else:
                words = [ word for word in self.model.vocab ]

        try:
            word_vectors = np.array([self.model[w] for w in words])
        except KeyError as e:
            print(e)
            return

        twodim = PCA(n_components=2).fit_transform(word_vectors)
        plt.figure(figsize=(6,6))
        plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
        for word, (x,y) in zip(words, twodim):
            print(word, (x,y))
            plt.text(x, y, word)
        plt.show()
        return words

    def get_vector(self, word):
        return self.model[word]
    
    def get_heat_map(self, words=None, sample=0):
        if not words:
            if sample > 0:
                words = np.random.choice(list(self.get_vocabulary()), sample)
            else:
                words = [word for word in self.model.vocab]
        
        try:
            word_vectors = [self.get_vector(w) for w in words]
        except KeyError as e:
            print(f'no word: ', e)
            return

        word_vectors = np.array(word_vectors)
        if word_vectors.shape[1] > 50:
            print('reducing dimensionality...')
            random_sample = random.sample(range(0, word_vectors.shape[1]), 50)
            random_sample = sorted(random_sample)
            word_vectors = word_vectors[:, random_sample]
            print(f'new shape vectors: {word_vectors.shape[1]}')

        plt.figure(figsize=(30,3))
        sns.heatmap(data=word_vectors, xticklabels=random_sample, yticklabels=words, cbar=True, vmin=-1, vmax=1, linewidths=0.7)
        plt.show()

        return words

def save_model(path):
    global w2v

    if not w2v:
        print('please train or load a model')
        return
    w2v.save_model(path)


def train_model(nameCorpus='cookbook',
                epochs=10,
                vector_size=300,
                min_count=1,
                window=10):
    global w2v

    try:
        print('loading corpus: {0}...\n'.format(nameCorpus))
        corpus = utils.load_corpus(nameCorpus)
        print('corpus already loaded\n')
    except NameError:
        print('the corpus {0} is not available'.format(nameCorpus))
        return

    corpus = list(corpus)
    if not corpus:
        print('The corpus is empty. There is nothing to train.')
        return

    w2v = Word2vec(vector_size, min_count, window)
    print('training... please wait\n')
    w2v.train(corpus, epochs=epochs)
    print('The model with {0} corpus is trained'.format(nameCorpus))
    return w2v.model.wv


def load_model(path):
    global w2v

    try:
        w2v = Word2vec(path=path)
    except FileNotFoundError:
        print("the vectors can't be founded")
        return
    return w2v.model.wv


def get_similar_words(word, number=10):
    global w2v

    if not w2v:
        print('please train or load a model')
        return

    try:
        most_similar = w2v.model.wv.similar_by_word(word, number)
    except KeyError as e:
        print(e)
        return

    return most_similar

def get_least_similar_words(word, number=10):
    global w2v

    if not w2v:
        print('please train or load a model')
        return

    try:
        least_similar = w2v.model.most_similar(negative=[word], topn=number)
    except KeyError as e:
        print(e)
        return

    return least_similar


def get_semantic_outlier(words: list):
    global w2v

    if not w2v:
        print('please train or load a model')
        return

    try:
        semantic_outlier = w2v.model.doesnt_match(words)
    except KeyError as e:
        print(e)
        return

    return semantic_outlier

def get_analogy(example: list, analogy: list, number: int = 1):
    global w2v

    if not w2v:
        print('please train or load a model')
        return

    try:
        analogy = w2v.model.most_similar(example, analogy, topn=number)
    except KeyError as e:
        print(e)
        return

    return analogy

def get_vocabulary():
    global w2v

    if not w2v:
        print('please train or load a model')
        return

    return w2v.get_vocabulary()


def cosine_similarity_between_words(word1, word2):
    global w2v

    if not w2v:
        print('please train or load a model')
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
        print('please train or load a model')
        return

    w2v.graphic_words(words, sample)


def get_vector(word):
    global w2v

    if not w2v:
        print('please train or load a model')
        return

    try:
        vector = w2v.get_vector(word)
    except KeyError as e:
        print(e)
        return
    return vector

def get_heatmap_words(words=None, sample=0):
    global w2v

    if not w2v:
        print('please train or load a model')
        return

    w2v.get_heat_map(words, sample)