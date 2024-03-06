import product.factor as factor
import product.product_operator as product_operator
import pandas as pd

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
        products = product_operator.PRODUCTS.keys()
    if factors is None:
        factors = factor.get_factor_dict(factor.REDUCED_EXPERIMENT_FACTOR_SIZES)

    product_dict = {product_name: product_operator.PRODUCTS[product_name] for product_name in products}

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