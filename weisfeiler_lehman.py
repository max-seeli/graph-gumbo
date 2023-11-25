import networkx as nx
import numpy as np


def embed_graph_wl(graph, iterations):
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
        return np.array([embed_graph_wl(g, iterations) for g in graph])
    
    return nx.weisfeiler_lehman_graph_hash(graph, iterations=iterations)
