# -*- coding: utf-8 -*-

"""
Create a graph model
"""
import argparse
import os
from paragraph.Graph import Graph
# pipeline: create_graph -> extract_features -> train -> evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Folder with sentence pais')
    parser.add_argument('output',
                        help='File to write graph model file (gpickle format)')
    args = parser.parse_args()

    files = os.listdir(args.input)
    if len(files) != 2:
        print("create_graph.py: error: The folder must to contains 2 files."
              "e.g: 'train-h' and 'train-t'. \n"
              "The first with 'h' sentences and the last with 't' sentences.")
        exit()

    G = Graph(input_h=args.input+files[0],
              input_t=args.input+files[1]).create_graph()

    Graph.export(G, args.output+'graphmodel.gpickle')
