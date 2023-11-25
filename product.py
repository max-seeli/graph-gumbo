import networkx as nx


def get_all_products(graph, factor_graph):
    """
    Get the Cartesian, strong and tensor products of a graph and a factor graph.
    Also works with a list of graphs.

    Parameters
    ----------
    graph : networkx.Graph, list of networkx.Graph
        The graph or graphs to get the products of.
    factor_graph : networkx.Graph
        The factor graph to get the products of.

    Returns
    -------
    c_graph : networkx.Graph, list of networkx.Graph
        The Cartesian product of the graph and the factor graph.
    s_graph : networkx.Graph, list of networkx.Graph
        The strong product of the graph and the factor graph.
    t_graph : networkx.Graph, list of networkx.Graph
        The tensor product of the graph and the factor graph.
    """
    if isinstance(graph, list):
        c_graph = []
        s_graph = []
        t_graph = []
        for g in graph:
            c, s, t = get_all_products(g, factor_graph)
            c_graph.append(c)
            s_graph.append(s)
            t_graph.append(t)
    else:
        c_graph = nx.cartesian_product(graph, factor_graph)
        s_graph = nx.strong_product(graph, factor_graph)
        t_graph = nx.tensor_product(graph, factor_graph)
    return c_graph, s_graph, t_graph
    