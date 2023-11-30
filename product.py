import networkx as nx
from itertools import product


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
    

def modular_product(G, H):
    """
    Get the modular product of two graphs.

    Definition
    ----------
    The vertex set of the modular product is the Cartesian product of the 
    vertex sets of the two graphs. For any two vertices `(u, v)` and `(u', v')`
    in the modular product, there is an edge between them if and only if
    `u` is different from `u'` and `v` is different from `v'` and either:
    - `u` and `u'` are adjacent in the first graph and `v` and `v'` are adjacent
        in the second graph, or
    - `u` and `u'` are not adjacent in the first graph and `v` and `v'` are not
        adjacent in the second graph.

    Parameters
    ----------
    G : networkx.Graph
        The first graph.
    H : networkx.Graph
        The second graph.

    Returns
    -------
    M : networkx.Graph
        The modular product of G and H.
    """
    M = nx.Graph()

    # Cartesian product of the vertex sets
    for (u, v) in product(G.nodes(), H.nodes()):
        M.add_node((u, v))

    # Add edges based on the conditions
    for (u, v) in M.nodes():
        for (u_prime, v_prime) in M.nodes():
            if u != u_prime and v != v_prime:
                condition1 = G.has_edge(u, u_prime) and H.has_edge(v, v_prime)
                condition2 = not G.has_edge(u, u_prime) and not H.has_edge(v, v_prime)
                if condition1 or condition2:
                    M.add_edge((u, v), (u_prime, v_prime))
    return M