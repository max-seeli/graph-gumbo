import networkx as nx
import matplotlib.pyplot as plt
import os

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

def save_pdf(fig, name):
    """
    Saves a figure in the pdf format at 'img/plots'.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    name : str
        The name of the file to save the figure to
    """
    if not os.path.exists('img/plots'):
        os.makedirs('img/plots')
    file_name = os.path.join('img/plots', name + '.pdf')

    fig.savefig(file_name, format='pdf', dpi=300, bbox_inches='tight',)