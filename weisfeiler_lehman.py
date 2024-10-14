import networkx as nx
import numpy as np


def wl_embedding(graph, iterations, node_attr=None, edge_attr=None):
    """
    Embed a graph into a hash. The embedding is the Weisfeiler-Lehman hash of the
    graph.

    Parameters
    ----------
    graph : networkx.Graph, list of networkx.Graph
        The graph to embed.
    iterations : int
        The number of WL iterations to perform.

    Returns
    -------
    embedding : string, list of string
        The embedding of the graph.
    """
    if type(graph) is list:
        # return numpy array of embeddings
        return np.array([wl_embedding(g, iterations, node_attr, edge_attr) for g in graph])

    return nx.weisfeiler_lehman_graph_hash(graph, iterations=iterations, node_attr=node_attr, edge_attr=edge_attr)
