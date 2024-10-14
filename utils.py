import os

import matplotlib.pyplot as plt
import networkx as nx


def results_to_latex(results_df):
    """
    Convert a results dataframe to a latex table.

    The latex table should have the following structure:
    - toprule
    - Graph Products
    - Each factor graph (with category rotatet 90 degrees)

    The graph products and factor graphs should be variable depending on the results_df.

    The results_df should have the following structure:
    - index: factor graph
    - columns: graph products
    """
    columns = results_df.columns
    index = results_df.index

    graph_types = {
        'K': 'Complete',
        'P': 'Path',
        'S': 'Star',
    }

    graphs_per_type = {v: [] for v in graph_types.values()}

    for factor_graph in index:
        graph_type = graph_types[factor_graph[0]]
        graphs_per_type[graph_type].append(factor_graph)

    best_results = results_df.idxmin(axis=0)

    latex = '\\begin{tabular}{c' + '|r' * len(columns) + 'r}\n'
    latex += '\t\\toprule\n'
    latex += '\t& \\multicolumn{' + \
        str(len(columns)) + '}{c}{Graph Products} & \\\\\n'
    multicol_columns = ['\\multicolumn{1}{c}{' + c + '}' for c in columns]
    latex += '\tFactor & ' + ' & '.join(multicol_columns) + ' & \\\\\n'

    for graph_type, factor_graphs in graphs_per_type.items():
        latex += '\t\\addlinespace[0.5ex]\n'
        latex += '\t\\cline{1-' + str(len(columns) + 2) + '}\n'
        latex += '\t\\addlinespace[0.5ex]\n'
        for i, factor_graph in enumerate(factor_graphs):

            str_results = []
            for product in columns:
                if factor_graph == best_results[product]:
                    str_results.append(
                        '\\textbf{' + str(results_df.loc[factor_graph, product]) + '}')
                else:
                    str_results.append(
                        str(results_df.loc[factor_graph, product]))

            latex += '\t$' + \
                factor_graph[0] + '_{' + factor_graph[1:] + \
                '}$ & ' + ' & '.join(str_results) + ' & '
            if i == 0:
                latex += '\\multirow{' + str(len(factor_graphs)) + \
                    '}{*}{\\rotatebox[origin=c]{270}{' + graph_type + '}} '
            latex += '\\\\\n'
    latex += '\t\\bottomrule\n'
    latex += '\\end{tabular}'

    with open('results.tex', 'w') as f:
        f.write(latex)


def prepend_dict(d, prepend):
    """
    Prepend a string to each key in a dictionary.

    Parameters
    ----------
    d : dict
        The dictionary to prepend to.
    prepend : str
        The string to prepend.

    Returns
    -------
    dict
        The dictionary with the prepended keys.
    """
    return {prepend + k: v for k, v in d.items()}


NODE_COLOR = 'lightblue'
NODE_SIZE = 1000


def plot_graph(graph, title=None, pos=None, figsize=(5, 5)):
    """
    Plot a graph using networkx and matplotlib.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to plot.
    title : str, optional
        The title of the plot.
    pos : dict, optional
        The positions of the nodes in the plot.
    figsize : tuple of int, optional
        The size of the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    nx.draw(graph, pos=pos, with_labels=True, ax=ax,
            node_color=NODE_COLOR,
            node_size=NODE_SIZE)
    if title:
        ax.set_title(title)
    plt.show()


def plot_horizontally(graphs, titles=None, pos=None, figsize=None):
    """
    Plot a list of graphs horizontally using networkx and matplotlib.

    Parameters
    ----------
    graphs : list of networkx.Graph
        The graphs to plot.
    titles : list of str, optional
        The titles of the plots.
    pos : dict, optional
        The positions of the nodes in the plots.
    figsize : tuple of int, optional
        The size of the plots.
    """
    if figsize is None:
        figsize = (5 * len(graphs), 5)
    fig, axes = plt.subplots(1, len(graphs), figsize=figsize)
    for i, graph in enumerate(graphs):
        nx.draw(graph, pos=pos, with_labels=True, ax=axes[i],
                node_color=NODE_COLOR,
                node_size=NODE_SIZE)
        if titles:
            axes[i].set_title(titles[i])
    plt.show()


def plot_collisions(collisions, graph_list, num_examples=3, fix_positions=False, contains_cycle=False):

    idx_pairs = list(collisions)

    if contains_cycle:
        for pair in idx_pairs:
            try:
                nx.find_cycle(graph_list[pair[0]])
                nx.find_cycle(graph_list[pair[1]])
            except nx.exception.NetworkXNoCycle:
                idx_pairs.remove(pair)

    for i in range(num_examples):
        if i >= len(idx_pairs):
            print("Warning: Not enough examples to plot")
            break

        graph_pair = [graph_list[idx_pairs[i][0]], graph_list[idx_pairs[i][1]]]

        pos = nx.spring_layout(graph_pair[0]) if fix_positions else None
        plot_horizontally(graph_pair, titles=[
                          'Graph 1', 'Graph 2'], figsize=(10, 5), pos=pos)


def save_pdf(fig, name):
    """
    Saves a figure in the pdf format at 'visual/plots'.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    name : str
        The name of the file to save the figure to
    """
    if not os.path.exists('visual/plots'):
        os.makedirs('visual/plots')
    file_name = os.path.join('visual/plots', name + '.pdf')

    fig.savefig(file_name, format='pdf', dpi=300, bbox_inches='tight',)
