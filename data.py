import numpy as np
from torch_geometric.datasets import TUDataset
import torch_geometric.utils as pyg_utils

def MUTAG():
    """
    Load the MUTAG dataset.
    
    Returns
    -------
    mutag_graphs : list of networkx.Graph
        The graphs in the dataset.
    """
    mutag = TUDataset(root='/tmp/MUTAG', name='MUTAG')

    dataset = get_dataset(mutag)

    for graph in dataset[0]:
        # Reduce the one-hot at node attribute 'x' to a single integer
        for node in graph.nodes:
            graph.nodes[node]['x'] = np.array(graph.nodes[node]['x']).argmax()
    return dataset 




def get_dataset(dataset):
    """
    Load a dataset from PyTorch Geometric repository in networkx format.

    Parameters
    ----------
    dataset : torch_geometric.datasets
        The dataset to load.

    Returns
    -------
    nx_graphs : list of networkx.Graph
        The graphs in the dataset.
    """
    nx_graphs = []
    for data in dataset:
        nx_graphs.append(
            pyg_utils.to_networkx(data, node_attrs=['x'], to_undirected=True)
        )
    return nx_graphs, dataset._data.y.numpy()
