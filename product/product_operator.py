from itertools import product

import networkx as nx
import numpy as np
import pandas as pd

from product import factor


def abstract_product(G, H, incidence_table):
    """
    Computes the graph product defined by the given incidence table.

    The incidence table defines the conditions under which an edge is created 
    between two vertices in the product graph. It's a 3x3 matrix where rows and columns
    represent the relationship between vertices in G and H (same vertex, an edge exists, 
    or no edge), and cells contain 1 (add an edge) or 0 (do not add an edge).

    This concept is a generalization of all the other graph products and was introduced
    by Imrich and Klavzar in 2000.

    Parameters
    ----------
    G : networkx.Graph
        The first input graph.
    H : networkx.Graph
        The second input graph.
    incidence_table : np.ndarray, shape (3, 3)
        A 3x3 matrix defining the edge addition rules based on vertex relationships.

    Returns
    -------
    networkx.Graph
        The resulting graph product of G and H based on the incidence table.
    """
    M = nx.Graph()

    # Determines the relationship between two vertices.
    def vertex_relation(graph, a, b):
        if a == b:
            return 0  # Vertices are the same.
        elif graph.has_edge(a, b):
            return 1  # An edge exists between the vertices.
        else:
            return 2  # No edge exists between the vertices.

    for u in G:
        for v in H:
            M.add_node((u, v))

    for (u1, v1), (u2, v2) in product(M.nodes, repeat=2):
        rel_G = vertex_relation(G, u1, u2)
        rel_H = vertex_relation(H, v1, v2)

        # Add an edge if the incidence table indicates so, excluding self-loops.
        if incidence_table[rel_G, rel_H] == 1 and (u1, v1) != (u2, v2):
            M.add_edge((u1, v1), (u2, v2))
    return M


def all_products(G, H):
    """
    Compute all possible graph products between two graphs.

    Parameters
    ----------
    G : networkx.Graph
        The first input graph.
    H : networkx.Graph
        The second input graph.

    Returns
    -------
    dict
        A dictionary containing all possible graph products between G and H.
    """
    # Iterate over all binary matrices of size 3x3.
    products = {}
    for i in range(2 ** 8):
        incidence_table = np.array([int(x) for x in f'0{i:08b}']).reshape(3, 3)
        products[f'Product {i}'] = abstract_product(G, H, incidence_table)
    return products


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

    def attribute_product(d1, d2): return {k: (
        d1.get(k), d2.get(k)) for k in set(d1) | set(d2)}
    # Cartesian product of the vertex sets
    for (u, v) in product(G.nodes(), H.nodes()):
        # Create cartesian product of all attributes if they exist
        M.add_node((u, v), **attribute_product(G.nodes[u], H.nodes[v]))

    # Add edges based on the conditions
    for (u, v) in M.nodes():
        for (u_prime, v_prime) in M.nodes():
            condition1 = G.has_edge(u, u_prime) and H.has_edge(v, v_prime)
            condition2 = (not G.has_edge(u, u_prime) and not H.has_edge(v, v_prime)
                          and u != u_prime and v != v_prime)
            if condition1:
                M.add_edge((u, v), (u_prime, v_prime), condition=0)
            elif condition2:
                M.add_edge((u, v), (u_prime, v_prime), condition=1)
    return M


def rooted_product_permutation_family(G, F):
    """
    Get the rooted product permutation family of two graphs.

    Definition
    ----------
    The rooted product permutation family of two graphs G and F is the sum of
    all rooted products of F and G with a root in G.

    Parameters
    ----------
    G : networkx.Graph
        The original graph.
    F : networkx.Graph
        The factor graph.

    Returns
    -------
    RPPF : networkx.Graph
        The rooted product permutation family of G and F.
    """
    RPPF = nx.Graph()

    for v in G.nodes():
        sum_RPPF = nx.rooted_product(F, G, root=v)
        RPPF = nx.disjoint_union(RPPF, sum_RPPF)
    return RPPF


PRODUCTS = {
    'Cartesian': nx.cartesian_product,
    'Strong': nx.strong_product,
    'Direct': nx.tensor_product,
    'Modular': modular_product,
    'Lexicographic': nx.lexicographic_product,
}


def generate_graph_product_table(graphs, products=None, factors=None, embedding=None, embedding_size=None):
    """
    Generate a table with a list of graph products for each combination of
    product operator and factor graph.

    Parameters
    ----------
    graphs : list of networkx.Graph
        The graphs to generate the products of.
    products : list of str or None (default)
        The products to generate (from the available products in the 
          product.product_operator.PRODUCTS dict). If None, all available
          products are generated.
    factors : dict {str: networkx.Graph} or None (default)
        The factor graphs to generate the products of. If None, all available
        factors are generated.
    embedding : GraphEmbedding or None (default)
        The embedding to use for the products. If None, the products are not
        embedded.
    embedding_size : int or None (default)
        The size of the embedding to use for the products. 

    Returns
    -------
    product_table : pd.DataFrame
        A table of the (embedded) products of the graphs and factor graphs.  
    """
    if products is None:
        products = PRODUCTS.keys()
    if factors is None:
        factors = factor.get_factor_dict(
            factor.REDUCED_EXPERIMENT_FACTOR_SIZES)

    product_dict = {product_name: PRODUCTS[product_name]
                    for product_name in products}

    product_table = pd.DataFrame(index=factors.keys(), columns=products)
    product_table.index.name = "Factor Graph"
    product_table.columns.name = "Graph Product"

    for factor_name, factor_graph in factors.items():
        for product_name, product_function in product_dict.items():
            transformed = []
            for graph in graphs:
                graph = product_function(graph, factor_graph)
                if embedding is not None:
                    transformed.append(embedding(graph, embedding_size))
                else:
                    transformed.append(graph)
            product_table.loc[factor_name, product_name] = transformed

    return product_table
