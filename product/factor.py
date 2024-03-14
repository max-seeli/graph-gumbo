import networkx as nx


__FACTORS = {
    "Complete": {
        "generator": nx.complete_graph,
        "short_name": "K{}",
    },
    "Path": {
        "generator": nx.path_graph,
        "short_name": "P{}",
    },
    "Star": {
        "generator": nx.star_graph,
        "short_name": "S{}",
    },
}

def get_factor_dict(sizes, factors=None, sep_self_loop_node=False):
    """
    Get a set of factor graphs of different sizes.
    
    Parameters
    ----------
    sizes : list of int
        The sizes of the factor graphs to generate.
    factors : list of str or None (default) 
        The factors to generate (from the available factors in the FACTORS dict).
          If None, all available factors are generated. 
    
    Returns
    -------
    factor_set : dict
        A dictionary mapping the short name of the factor graph to the factor graph.
    """
    if factors is None:
        factors = __FACTORS.keys()
    factor_set = {__FACTORS[f]["short_name"].format(n): __FACTORS[f]["generator"](n) for f in factors for n in sizes}
    
    if sep_self_loop_node:
        for k, v in factor_set.items():
            non_existing_node = "S"
            v.add_edge("S", "S")
            factor_set[k] = v
    
    return factor_set

FACTORS = {k: v["generator"] for k, v in __FACTORS.items()}

EXPERIMENT_FACTOR_SIZES = [3, 5, 7, 9, 11, 13, 15]
REDUCED_EXPERIMENT_FACTOR_SIZES = [3, 5, 7, 13]
