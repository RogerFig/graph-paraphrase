import numpy as np
import networkx as nx

from nltk import tokenize
from nltk.corpus import stopwords
import util
from regularizacao import *

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class GenerateGraph:

    def __init__(self, train_input_h, train_input_t, test_input_h, test_input_t):
        self.train_input_h = self.read_file(train_input_h)
        self.train_input_t = self.read_file(train_input_t)
        self.test_input_h = self.read_file(test_input_h)
        self.test_input_t = self.read_file(test_input_t)

    
    @staticmethod
    def read_file(input_f):
        return open(input_f, 'r',).readlines()

    @staticmethod
    def preprocess(snt_h, snt_t):
        tokens_h = tokenize.word_tokenize(snt_h, language='portuguese')
        tokens_t = tokenize.word_tokenize(snt_t, language='portuguese')
        return [t.lower() for t in tokens_h if t not in stopwords.words(u'portuguese')], \
               [t.lower() for t in tokens_t if t not in stopwords.words(u'portuguese')]

    def get_sentences(self):
        train_sentences_h, train_sentences_t = [], []
        for snt_h, snt_t in zip(self.train_input_h, self.train_input_t):
            tokens_h, tokens_t = self.preprocess(snt_h, snt_t)
            train_sentences_h.append(tokens_h)
            train_sentences_t.append(tokens_h)

        test_sentences_h, test_sentences_t = [], []
        for snt_h, snt_t in zip(self.test_input_h, self.test_input_t):
            tokens_h, tokens_t = self.preprocess(snt_h, snt_t)
            test_sentences_h.append(tokens_h)
            test_sentences_t.append(tokens_h)

        return train_sentences_h, train_sentences_t, test_sentences_h, test_sentences_t


    def create_graph(self):
        train_sentences_h, train_sentences_t, test_sentences_h, test_sentences_t = self.get_sentences()
        
        G = nx.Graph()

        dict_token_nodes = {}
        nodes_ids = 0
        node_sentences_pair = {}

        ## Train ###
        for id, sentence_pair in enumerate(zip(train_sentences_h, train_sentences_t)):
            chave = 'train_%d'%id
            node_sentences_pair[chave] = nodes_ids
            nodes_ids+=1
            G.add_node(node_sentences_pair[chave], type='sentence_pair', value=chave)

            for token in sentence_pair[0]:
                if token not in dict_token_nodes:
                    dict_token_nodes[token] = nodes_ids
                    nodes_ids+=1
                    G.add_node(dict_token_nodes[token], type='token', value=token)
                    G.add_edge(node_sentences_pair[chave],dict_token_nodes[token])
                else:
                    G.add_edge(node_sentences_pair[chave],dict_token_nodes[token])

            for token in sentence_pair[1]:
                if token not in dict_token_nodes:
                    dict_token_nodes[token] = nodes_ids
                    nodes_ids+=1
                    G.add_node(dict_token_nodes[token], type='token', value=token)
                    G.add_edge(node_sentences_pair[chave],dict_token_nodes[token])
                else:
                    G.add_edge(node_sentences_pair[chave],dict_token_nodes[token])

        ## Test ###
        for id, sentence_pair in enumerate(zip(test_sentences_h, test_sentences_t)):
            chave = 'test_%d'%id
            node_sentences_pair[chave] = nodes_ids
            nodes_ids+=1
            G.add_node(node_sentences_pair[chave], type='sentence_pair', value=chave)

            for token in sentence_pair[0]:
                if token not in dict_token_nodes:
                    dict_token_nodes[token] = nodes_ids
                    nodes_ids+=1
                    G.add_node(dict_token_nodes[token], type='token', value=token)
                    G.add_edge(node_sentences_pair[chave],dict_token_nodes[token])
                else:
                    G.add_edge(node_sentences_pair[chave],dict_token_nodes[token])
                    
            for token in sentence_pair[1]:
                if token not in dict_token_nodes:
                    dict_token_nodes[token] = nodes_ids
                    nodes_ids+=1
                    G.add_node(dict_token_nodes[token], type='token', value=token)
                    G.add_edge(node_sentences_pair[chave],dict_token_nodes[token])
                else:
                    G.add_edge(node_sentences_pair[chave],dict_token_nodes[token])

        return G, node_sentences_pair

if __name__ == '__main__':

    print('### Tokenizing and Preprocessing ###')
    G, node_sentences_pair = GenerateGraph(train_input_h='corpora/assin+msr/train/train-h.txt',
                                 train_input_t='corpora/assin+msr/train/train-t.txt', test_input_h='corpora/assin+msr/test/test-h.txt',
                                 test_input_t='corpora/assin+msr/test/test-t.txt',).create_graph()

    print(len(node_sentences_pair))
    train_labels = util.get_labels('corpora/assin+msr/labels-train.txt')
    test_labels = util.get_labels('corpora/assin+msr/labels-test.txt')
    reg = Regularization()
    reg.regulariza(G, node_sentences_pair, train_labels, '' ,method='gnetmine')
    
    df_train = pd.read_csv("features_2100_pre_anotados_train.csv")
    df_test = pd.read_csv("features_2100_pre_anotados_test.csv")

    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(20,20), random_state=42, max_iter=1000)
    clf.fit(df_train[['p','nao_p']], np.ravel(df_train[['classe']]))
    print('Training: ', clf.score(df_test[['p','nao_p']], test_labels))
    y_pred = clf.predict(df_test[['p','nao_p']])

    holdout_score = classification_report(test_labels, y_pred)
    print(holdout_score)

    