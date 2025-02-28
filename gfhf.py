# -*- coding: utf-8 -*-
# Fonte: https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.node_classification.hmn.harmonic_function.html#networkx.algorithms.node_classification.hmn.harmonic_function
# Author: Yuto Yamaguchi <yuto.ymgc@gmail.com>
"""Function for computing Harmonic function algorithm by Zhu et al.

References
----------
Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
Semi-supervised learning using gaussian fields and harmonic functions.
In ICML (Vol. 3, pp. 912-919).
"""
import networkx as nx
from networkx.algorithms import node_classification

from networkx.utils.decorators import not_implemented_for
from networkx.algorithms.node_classification.utils import (
    _get_label_info,
    _init_label_matrix,
    _propagate,
    _predict,
)

__all__ = ['harmonic_function']


@not_implemented_for('directed')
def harmonic_function(G, max_iter=30, label_name='label'):
    """Node classification by Harmonic function

    Parameters
    ----------
    G : NetworkX Graph
    max_iter : int
        maximum number of iterations allowed
    label_name : string
        name of target labels to predict

    Raises
    ----------
    `NetworkXError` if no nodes on `G` has `label_name`.

    Returns
    ----------
    predicted : array, shape = [n_samples]
        Array of predicted labels

    Examples
    --------
    >>> from networkx.algorithms import node_classification
    >>> G = nx.path_graph(4)
    >>> G.node[0]['label'] = 'A'
    >>> G.node[3]['label'] = 'B'
    >>> G.nodes(data=True)
    NodeDataView({0: {'label': 'A'}, 1: {}, 2: {}, 3: {'label': 'B'}})
    >>> G.edges()
    EdgeView([(0, 1), (1, 2), (2, 3)])
    >>> predicted = node_classification.harmonic_function(G)
    >>> predicted
    ['A', 'A', 'B', 'B']

    References
    ----------
    Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
    Semi-supervised learning using gaussian fields and harmonic functions.
    In ICML (Vol. 3, pp. 912-919).
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError(
            "harmonic_function() requires numpy: http://scipy.org/ ")
    try:
        from scipy import sparse
    except ImportError:
        raise ImportError(
            "harmonic_function() requires scipy: http://scipy.org/ ")

    def _build_propagation_matrix(X, labels):
        """Build propagation matrix of Harmonic function

        Parameters
        ----------
        X : scipy sparse matrix, shape = [n_samples, n_samples]
            Adjacency matrix
        labels : array, shape = [n_samples, 2]
            Array of pairs of node id and label id

        Returns
        ----------
        P : scipy sparse matrix, shape = [n_samples, n_samples]
            Propagation matrix

        """
        degrees = X.sum(axis=0).A[0]
        degrees[degrees == 0] = 1  # Avoid division by 0
        D = sparse.diags((1.0 / degrees), offsets=0)
        P = D.dot(X).tolil()
        P[labels[:, 0]] = 0  # labels[:, 0] indicates IDs of labeled nodes
        return P

    def _build_base_matrix(X, labels, n_classes):
        """Build base matrix of Harmonic function

        Parameters
        ----------
        X : scipy sparse matrix, shape = [n_samples, n_samples]
            Adjacency matrix
        labels : array, shape = [n_samples, 2]
            Array of pairs of node id and label id
        n_classes : integer
            The number of classes (distinct labels) on the input graph

        Returns
        ----------
        B : array, shape = [n_samples, n_classes]
            Base matrix
        """
        n_samples = X.shape[0]
        B = np.zeros((n_samples, n_classes))
        B[labels[:, 0], labels[:, 1]] = 1
        return B

    X = nx.to_scipy_sparse_matrix(G)  # adjacency matrix
    labels, label_dict = _get_label_info(G, label_name)

    if labels.shape[0] == 0:
        raise nx.NetworkXError(
            "No node on the input graph is labeled by '" + label_name + "'.")

    n_samples = X.shape[0]
    n_classes = label_dict.shape[0]

    F = _init_label_matrix(n_samples, n_classes)

    P = _build_propagation_matrix(X, labels)
    B = _build_base_matrix(X, labels, n_classes)

    remaining_iter = max_iter
    while remaining_iter > 0:
        F = _propagate(P, F, B)
        remaining_iter -= 1

    #predicted = _predict(F, label_dict)

    return F



def setup_module(module):
    """Fixture for nose tests."""
    from nose import SkipTest
    try:
        import numpy
    except ImportError:
        raise SkipTest("NumPy not available")
    try:
        import scipy
    except ImportError:
        raise SkipTest("SciPy not available")

if __name__ == '__main__':
    G=nx.Graph()
    G.add_nodes_from(range(1,16))
    G.node[2]['label'] = 'A'
    G.node[11]['label'] = 'B'
    G.add_edges_from([(1,3),(3,2),(3,4),(4,6),(4,12),(4,14),(6,5),(12,10),(12,15),(10,7),(10,8),(10,9),(10,11),(10,13)])
	# Se quiser usar a função da networkx para rotular logo basta usar:
    predicted = node_classification.harmonic_function(G)
    print(predicted)
    # Se quiser os valores de pertinência em cada classe deve-se usar a
    # função deste arquivo
    F = harmonic_function(G)
    print(F)