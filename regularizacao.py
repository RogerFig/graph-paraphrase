import networkx as nx
import numpy as np
from gfhf import harmonic_function
from llgc import local_and_global_consistency
from GNetMine import GNetMine
import zipfile
import ast
from random import shuffle as shuffle_list
import pandas as pd
import time

class Regularization:

    def __init__(self):
        pass

    def regulariza(self, G, sentence_nodes, train_labels, path_out, total_pre_anotados=0.3, method='gfhf'):
        """
            gerar as features de regularização
        """
        total_samples = len(train_labels)
        cods = list(range(total_samples))
        shuffle_list(cods)

        anotados = cods[0:int(total_samples*total_pre_anotados)]

        train = 'train_%d'
        test = 'test_%d'
        
        keys_anotados = []
        for i in range(total_samples):
            if i in anotados:
                G.nodes[sentence_nodes[train%i]]['label'] = train_labels[i]
                keys_anotados.append(train%i)

        with open('anotados.txt','w') as f:
            for i in anotados:
                f.write('%d\n'%i)

        labels = ['p','nao_p']
        columns = ['id',labels[0],labels[1],'classe']
        rows_train = []
        rows_test = []

        if method == 'gfhf':
            F = harmonic_function(G)
        elif method == 'llgc':
            F = local_and_global_consistency(G)

        if method in ['gfhf','llgc']:
            for key in sentence_nodes.keys():
                id_node = sentence_nodes[key]
                split_key_node = key.split('_')
                t = split_key_node[0]
                cod = int(split_key_node[1])
                if t == 'train':
                    rows_train.append([id_node,F[id_node][0],F[id_node][1],train_labels[cod]])

                if t == 'test':
                    rows_test.append([id_node,F[id_node][0],F[id_node][1]])
        elif method == 'gnetmine':
            M = GNetMine(graph = G)
            c = M.run()
            F = M.f['sentence_pair']
            labels = M.labels
            nodes = M.nodes_type['sentence_pair']
            dict_nodes = {k:i for i,k in enumerate(nodes)}
            for key in sentence_nodes.keys():
                id_node = sentence_nodes[key]
                split_key_node = key.split('_')
                t = split_key_node[0]
                cod = int(split_key_node[1])
                if t == 'train':
                    rows_train.append([id_node,F[dict_nodes[id_node]][0],F[dict_nodes[id_node]][1],train_labels[cod]])

                if t == 'test':
                    rows_test.append([id_node,F[dict_nodes[id_node]][0],F[dict_nodes[id_node]][1]])
        


        file_name_train = 'features_%s_pre_anotados_train.csv' % len(anotados)
        file_name_test = 'features_%s_pre_anotados_test.csv' % len(anotados)
        df_train = pd.DataFrame(rows_train, columns=columns)
        df_test = pd.DataFrame(rows_test, columns=columns[:3])
        df_train.to_csv(path_out+file_name_train,index=False)
        df_test.to_csv(path_out+file_name_test,index=False)