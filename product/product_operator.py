import networkx as nx
from itertools import product


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
            condition1 = G.has_edge(u, u_prime) and H.has_edge(v, v_prime)
            condition2 = (not G.has_edge(u, u_prime) and not H.has_edge(v, v_prime) 
                          and u != u_prime and v != v_prime)
            if condition1:
                M.add_edge((u, v), (u_prime, v_prime), condition = 0)
            elif condition2:
                M.add_edge((u, v), (u_prime, v_prime), condition = 1)
    return M

PRODUCTS = {
    'Cartesian': nx.cartesian_product,
    'Strong': nx.strong_product,
    'Tensor': nx.tensor_product,
    'Modular': modular_product,
    'Lexicographic': nx.lexicographic_product,
}
