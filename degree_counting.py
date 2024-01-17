import networkx as nx
import numpy as np

def embed_graph_degree_sequence(graph, size):
    """
    Embed a graph into a vector. The vector is of size `size`. The embedding is
    the embedding of the degree sequence of the graph.

    Parameters
    ----------
    graph : networkx.Graph, list of networkx.Graph
        The graph to embed.
    size : int
        The size of the vector to embed into.
    
    Returns
    -------
    embedding : numpy.ndarray
        The embedding of the graph.
    """
    if type(graph) is list:
        return np.array([embed_graph_degree_sequence(g, size) for g in graph])
    
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)

    if len(degree_sequence) > size:
        print("Warning: the degree sequence of the graph has more elements than the embedding size. The degree sequence will be truncated.")
        degree_sequence = degree_sequence[:size]
    else:
        pad_length = size - len(degree_sequence)
        degree_sequence = degree_sequence + [0] * pad_length
    return np.array(degree_sequence, dtype=np.int32)

def embed_graph_degrees(graph, size):
    """
    Embed a graph into a vector. The vector is of size `size`. The embedding is
    the embedding of the degree counts of the graph.

    Parameters
    ----------
    graph : networkx.Graph, list of networkx.Graph
        The graph to embed.
    size : int
        The size of the vector to embed into.
    
    Returns
    -------
    embedding : numpy.ndarray
        The embedding of the graph.
    """
    if type(graph) is list:
        return np.array([embed_graph_degrees(g, size) for g in graph])
    
    degree_counts = count_degrees(graph)
    return embed_degree_counts(degree_counts, size)

def count_degrees(graph):
    """
    Count the number of nodes with each specific node degree.
    
    Parameters
    ----------
    graph : networkx.Graph
        The graph to count.
    
    Returns
    -------
    degree_counts : dict
        A dictionary mapping degrees to the number of nodes with the degree.
    """
    degree_counts = {}

    for node in graph.nodes():
        degree = graph.degree(node)
        if degree not in degree_counts:
            degree_counts[degree] = 0
        degree_counts[degree] += 1

    return degree_counts
    
def embed_degree_counts(degree_counts, size):
    """
    Embed the degree counts for a graph into a vector. The vector is of size `size`.
    The index of the vector corresponds to the degree of the node count. The last
    index of the vector corresponds to the number of nodes with a degree greater or equal
    than `size`.

    Parameters
    ----------
    degree_counts : dict
        A dictionary mapping degrees to the number of nodes with that degree.
    size : int
        The size of the vector to embed into.
    
    Returns
    -------
    embedding : numpy.ndarray
        The embedding of the degree counts.
    """
    embedding = np.zeros(size, dtype=np.int32)
    for degree, count in degree_counts.items():
        if degree < size:
            embedding[degree] = count
        else:
            embedding[-1] += count
    if embedding[-1] > 0:
        print("Warning: {} nodes with degree greater or equal than {} were found. Increase the embedding size to avoid this.".format(embedding[-1], size))
    return embedding