import numpy as np
import networkx as nx

from nltk import tokenize
from nltk.corpus import stopwords
# import . import util
# from .regularizacao import *

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


class Graph:

    def __init__(self, input_h, input_t, G=None):
        self.train_input_h = self.read_file(input_h)
        self.train_input_t = self.read_file(input_t)
        if G is not None:
            self.G = G
        # self.test_input_h = self.read_file(test_input_h)
        # self.test_input_t = self.read_file(test_input_t)

    @classmethod
    def fromfilename(cls, input_h, input_t, name):
        return cls(input_h, input_t, nx.read_gpickle(name))

    @staticmethod
    def read_file(input_f):
        return open(input_f, 'r',).readlines()

    @staticmethod
    def preprocess(snt_h, snt_t):
        tokens_h = tokenize.word_tokenize(snt_h, language='portuguese')
        tokens_t = tokenize.word_tokenize(snt_t, language='portuguese')
        return [t.lower() for t in tokens_h if t not in stopwords.words(u'portuguese')], \
               [t.lower()
                for t in tokens_t if t not in stopwords.words(u'portuguese')]

    @staticmethod
    def export(G, outfile):
        nx.write_gpickle(G, outfile)

    def get_sentences(self):
        train_sentences_h, train_sentences_t = [], []
        for snt_h, snt_t in zip(self.train_input_h, self.train_input_t):
            tokens_h, tokens_t = self.preprocess(snt_h, snt_t)
            train_sentences_h.append(tokens_h)
            train_sentences_t.append(tokens_t)

        # test_sentences_h, test_sentences_t = [], []
        # for snt_h, snt_t in zip(self.test_input_h, self.test_input_t):
        #     tokens_h, tokens_t = self.preprocess(snt_h, snt_t)
        #     test_sentences_h.append(tokens_h)
        #     test_sentences_t.append(tokens_t)

        # , test_sentences_h, test_sentences_t
        return train_sentences_h, train_sentences_t

    def create_graph(self):
        # train_sentences_h, train_sentences_t, test_sentences_h, test_sentences_t = self.get_sentences()
        train_sentences_h, train_sentences_t = self.get_sentences()

        G = nx.Graph()

        dict_token_nodes = {}
        nodes_ids = 0
        node_sentences_pair = {}

        ## Train ###
        for id, sentence_pair in enumerate(zip(train_sentences_h, train_sentences_t)):
            chave = 'train_%d' % id
            node_sentences_pair[chave] = nodes_ids
            nodes_ids += 1
            G.add_node(node_sentences_pair[chave],
                       type='sentence_pair', value=chave)

            for token in sentence_pair[0]:
                if token not in dict_token_nodes:
                    dict_token_nodes[token] = nodes_ids
                    nodes_ids += 1
                    G.add_node(dict_token_nodes[token],
                               type='token', value=token)
                    G.add_edge(
                        node_sentences_pair[chave], dict_token_nodes[token])
                else:
                    G.add_edge(
                        node_sentences_pair[chave], dict_token_nodes[token])

            for token in sentence_pair[1]:
                if token not in dict_token_nodes:
                    dict_token_nodes[token] = nodes_ids
                    nodes_ids += 1
                    G.add_node(dict_token_nodes[token],
                               type='token', value=token)
                    G.add_edge(
                        node_sentences_pair[chave], dict_token_nodes[token])
                else:
                    G.add_edge(
                        node_sentences_pair[chave], dict_token_nodes[token])

        ## Test ###
        # for id, sentence_pair in enumerate(zip(test_sentences_h, test_sentences_t)):
        #     chave = 'test_%d' % id
        #     node_sentences_pair[chave] = nodes_ids
        #     nodes_ids += 1
        #     G.add_node(node_sentences_pair[chave],
        #                type='sentence_pair', value=chave)

        #     for token in sentence_pair[0]:
        #         if token not in dict_token_nodes:
        #             dict_token_nodes[token] = nodes_ids
        #             nodes_ids += 1
        #             G.add_node(dict_token_nodes[token],
        #                        type='token', value=token)
        #             G.add_edge(
        #                 node_sentences_pair[chave], dict_token_nodes[token])
        #         else:
        #             G.add_edge(
        #                 node_sentences_pair[chave], dict_token_nodes[token])

        #     for token in sentence_pair[1]:
        #         if token not in dict_token_nodes:
        #             dict_token_nodes[token] = nodes_ids
        #             nodes_ids += 1
        #             G.add_node(dict_token_nodes[token],
        #                        type='token', value=token)
        #             G.add_edge(
        #                 node_sentences_pair[chave], dict_token_nodes[token])
        #         else:
        #             G.add_edge(
        #                 node_sentences_pair[chave], dict_token_nodes[token])

        return G


def expand_graph(self):
        # train_sentences_h, train_sentences_t, test_sentences_h, test_sentences_t = self.get_sentences()
    sentences_h, sentences_t = self.get_sentences()

    dict_token_nodes = {}
    nodes_ids = 0
    node_sentences_pair = {}

    ## Train ###
    for id, sentence_pair in enumerate(zip(sentences_h, sentences_t)):
        chave = 'train_%d' % id
        node_sentences_pair[chave] = nodes_ids
        nodes_ids += 1
        G.add_node(node_sentences_pair[chave],
                   type='sentence_pair', value=chave)

        for token in sentence_pair[0]:
            if token not in dict_token_nodes:
                dict_token_nodes[token] = nodes_ids
                nodes_ids += 1
                G.add_node(dict_token_nodes[token],
                           type='token', value=token)
                G.add_edge(
                    node_sentences_pair[chave], dict_token_nodes[token])
            else:
                G.add_edge(
                    node_sentences_pair[chave], dict_token_nodes[token])

        for token in sentence_pair[1]:
            if token not in dict_token_nodes:
                dict_token_nodes[token] = nodes_ids
                nodes_ids += 1
                G.add_node(dict_token_nodes[token],
                           type='token', value=token)
                G.add_edge(
                    node_sentences_pair[chave], dict_token_nodes[token])
            else:
                G.add_edge(
                    node_sentences_pair[chave], dict_token_nodes[token])

    return G


if __name__ == '__main__':

    print('### Tokenizing and Preprocessing ###')
    # Obsoleto
    G, node_sentences_pair = Graph(input_h='corpora/assin+msr/train/train-h.txt',
                                   input_t='corpora/assin+msr/train/train-t.txt',
                                   test_input_h='corpora/assin+msr/test/test-h.txt',
                                   test_input_t='corpora/assin+msr/test/test-t.txt',).create_graph()

    print(len(node_sentences_pair))
    nx.write_gpickle(G, "graph_model1.gpickle")
    # train_labels = util.get_labels('corpora/assin+msr/labels-train.txt')
    # test_labels = util.get_labels('corpora/assin+msr/labels-test.txt')
    # reg = Regularization()
    # reg.regulariza(G, node_sentences_pair, train_labels, '',
    #                total_pre_anotados=0.4, method='llgc')

    # df_train = pd.read_csv("features_2800_pre_anotados_train.csv")
    # df_test = pd.read_csv("features_2800_pre_anotados_test.csv")

    # clf = MLPClassifier(solver='adam', hidden_layer_sizes=(20, 20), random_state=42, max_iter=1000)
    # clf.fit(df_train[['p', 'nao_p']], np.ravel(df_train[['classe']]))
    # print('Training: ', clf.score(df_test[['p', 'nao_p']], test_labels))
    # y_pred = clf.predict(df_test[['p', 'nao_p']])

    # holdout_score = classification_report(test_labels, y_pred)
    # print(holdout_score)
