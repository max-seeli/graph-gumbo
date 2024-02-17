import networkx as nx
import numpy as np

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

def count_degree_sequence(graph):
    """
    Given a graph, return a count-like representation of the degree sequence.
    
    Parameters
    ----------
    graph : networkx.Graph
        The graph to count.
    
    Returns
    -------
    degree_sequence_counts : dict
        A dictionary mapping positions in the degree sequence to the degree
        at that position
    """
    degree_sequence = sorted([d for _, d in graph.degree()], reverse=True)
    degree_sequence_counts = {k: v for k, v in enumerate(degree_sequence)}
    return degree_sequence_counts


operations = {
    "basis_cycle": {
        "name": "Cycle basis",
        "count_function": count_cycle_sizes,
        "overflow_idx": 0,
    },
    "degree": {
        "name": "Degree",
        "count_function": count_degrees,
        "overflow_idx": -1,
    },
    "degree_sequence": {
        "name": "Degree sequence",
        "count_function": count_degree_sequence,
        "overflow_idx": None,
    },
}

def embed_graph(graph, size, operation):
    """
    Embed a graph into a vector. The vector is of size `size`. The embedding is
    the embedding of the counts of the graph using the specified operation.

    Parameters
    ----------
    graph : networkx.Graph, list of networkx.Graph
        The graph to embed.
    size : int
        The size of the vector to embed into.
    operation : str {"basis_cycle", "degree"}
        The operation to use to count the graph.
    
    Returns
    -------
    embedding : numpy.ndarray
        The embedding of the graph.
    """
    if type(graph) is list:
        return np.array([embed_graph(g, size, operation) for g in graph])
    
    op = operations[operation]
    counts = op["count_function"](graph)
    return embed_counts(counts, size, overflow_idx=op["overflow_idx"])


def embed_counts(counts, size, overflow_idx=None, warn=True):
    """
    Embed the counts of a specific key (e.g. degrees or cycles in a graph)
    into a vector. The vector is of size `size`. All counts of indices greater
    than `size` are summed into the `overflow_idx`. After that, the index of
    the vector corresponds to the key of the counts.

    Parameters
    ----------
    counts : dict(int, int)
        A dictionary mapping keys to the number of counts of that key.
    size : int
        The size of the vector to embed into.
    overflow_idx : int, optional (default=None)
        The index of the vector to sum the counts of cycles with length greater
        than `size`. If None, the counts of cycles with length greater than
        `size` are discarded.
    warn : bool
        If True, warn when counts of cycles with length greater than `size` are
        found.
    
    Returns
    -------
    embedding : numpy.ndarray
        The embedding of the cycle counts.
    """
    embedding = np.zeros(size, dtype=np.int32)

    for key, count in counts.items():
        if key < size:
            embedding[key] = count
        elif overflow_idx is not None:
            embedding[overflow_idx] += count
        elif warn:
                print("Warning: a count of key {} was found, but the embedding size is {}. Increase the embedding size to avoid this.".format(key, size))

    if overflow_idx is not None and warn and embedding[overflow_idx] > 0:
        print("Warning: {} counts of keys greater than {} were found. Increase the embedding size to avoid this.".format(embedding[overflow_idx], size))
    return embedding