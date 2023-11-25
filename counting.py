import networkx as nx
import numpy as np

def embed_graph_cycles(graph, size):
    """
    Embed a graph into a vector. The vector is of size `size`. The embedding is
    the embedding of the cycle counts of the graph.

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
        return [embed_graph_cycles(g, size) for g in graph]
    
    cycle_counts = count_cycle_sizes(graph)
    return embed_cycle_counts(cycle_counts, size)

def count_cycle_sizes(graph):
    """
    Count the number of cycles of each length for a graph.
    
    Parameters
    ----------
    graph : networkx.Graph
        The graph to count.
    
    Returns
    -------
    cycle_counts : dict
        A dictionary mapping cycle lengths to the number of cycles of that length.
    """
    cycle_counts = {}
    for cycle in nx.cycle_basis(graph):
        cycle_length = len(cycle)
        if cycle_length not in cycle_counts:
            cycle_counts[cycle_length] = 0
        cycle_counts[cycle_length] += 1
    return cycle_counts

def embed_cycle_counts(cycle_counts, size):
    """
    Embed the cycle counts for a graph into a vector. The vector is of size `size`.
    All counts of cycles with greater length than `size` are summed into the 0th
    element of the vector. After that, the index of the vector corresponds to the
    length of the cycle.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to embed.
    cycle_counts : dict
        A dictionary mapping cycle lengths to the number of cycles of that length.
    size : int
        The size of the vector to embed into.
    
    Returns
    -------
    embedding : numpy.ndarray
        The embedding of the cycle counts.
    """
    embedding = np.zeros(size)
    for cycle_length, count in cycle_counts.items():
        if cycle_length < size:
            embedding[cycle_length] = count
        else:
            embedding[0] += count
    return embedding