import numpy as np

from gensim.models import KeyedVectors
from nltk import tokenize
from nltk.corpus import stopwords
from scipy import spatial


class ExtractFeatures:

    def __init__(self, model, input_h, input_t):
        self.model = KeyedVectors.load_word2vec_format(model)
        self.input_h = self.read_file(input_h)
        self.input_t = self.read_file(input_t)

    @staticmethod
    def read_file(input_f):
        return open(input_f, 'r',).readlines()

    @staticmethod
    def preprocess(snt_h, snt_t):
        tokens_h = tokenize.word_tokenize(snt_h, language='portuguese')
        tokens_t = tokenize.word_tokenize(snt_t, language='portuguese')
        return [t.lower() for t in tokens_h if t not in stopwords.words(u'portuguese')], \
               [t.lower() for t in tokens_t if t not in stopwords.words(u'portuguese')]

    def filter_tokens(self, tokens_h, tokens_t):
        filtered_h, filtered_t = [], []
        for token in tokens_h:
            if token in self.model.vocab:
                filtered_h.append(token)
        for token in tokens_t:
            if token in self.model.vocab:
                filtered_t.append(token)
        return filtered_h, filtered_t

    def word_move_distance(self, tokens_h, tokens_t):
        return self.model.wmdistance(tokens_h, tokens_t)

    def cosine_distance_embeddings(self, tokens_h, tokens_t):
        vector_h = np.mean([self.model[token] for token in tokens_h], axis=0)
        vector_t = np.mean([self.model[token] for token in tokens_t], axis=0)
        return 1 - spatial.distance.cosine(vector_h, vector_t)

    def extract_features(self):
        features = []
        for snt_h, snt_t in zip(self.input_h, self.input_t):
            aux = []
            tokens_h, tokens_t = self.preprocess(snt_h, snt_t)
            tokens_h, tokens_t = self.filter_tokens(tokens_h, tokens_t)
            aux.append(float(self.word_move_distance(tokens_h, tokens_t)))
            aux.append(float(self.cosine_distance_embeddings(tokens_h, tokens_t)))
            features.append(aux)
        return features, features
