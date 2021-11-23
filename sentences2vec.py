import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns


from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
random.seed = 43

def get_vector_sentence(sentence: str):
    global model
    return model.encode(sentence)


def cosine_similarity_between_sentences(sentence1: str, sentence2: str):
    return util.dot_score(get_vector_sentence(sentence1), get_vector_sentence(sentence2))


def graphic_sentences(sentences: list):
    if len(sentences) < 2:
        print('please, add more sentences')
        return

    try:
        sentences_vec = np.array([get_vector_sentence(sent) for sent in sentences])
    except KeyError as e:
        print(e)
        return

    twodim = PCA(n_components=2).fit_transform(sentences_vec)
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for sent, (x,y) in zip(sentences, twodim):
        print(sent, (x,y))
        plt.text(x, y, sent)
    plt.show()
    return sent


def get_heat_map_sentences(sentences: list):
        sent_vectors = np.array([get_vector_sentence(sent) for sent in sentences])

        if sent_vectors.shape[1] > 50:
            print('reducing dimensionality...')
            random_sample = random.sample(range(0, sent_vectors.shape[1]), 100)
            random_sample = sorted(random_sample)
            sent_vectors = sent_vectors[:, random_sample]
            print(f'new shape vectors: {sent_vectors.shape[1]}')

        plt.figure(figsize=(30,3))
        sns.heatmap(data=sent_vectors, xticklabels=random_sample, yticklabels=sentences, cbar=True, vmin=-.5, vmax=.5, linewidths=0.7)
        plt.show()
        return sentences