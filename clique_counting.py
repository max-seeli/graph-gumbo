import networkx as nx

def count_cliques(graph, k):
    """
    Count the number of cliques of size k in a graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to count.
    k : int
        The size of the cliques to count.

    Returns
    -------
    count : int
        The number of cliques of size k in the graph.
    """
    return sum(1 for clique in nx.enumerate_all_cliques(graph) if len(clique) == k)

print(count_cliques(nx.complete_graph(5), 3))