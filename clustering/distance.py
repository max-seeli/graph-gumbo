import networkx as nx
import numpy as np
from scipy.spatial import distance as dist
from counting import BasisCycleEmbedding
from product.product_operator import modular_product
import os
from tqdm import tqdm
from itertools import combinations
from clustering.data.data_generator import read_graphs

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
embedder = BasisCycleEmbedding(size = 100)

def to_vec(graph):
    """
    Converts the graph to a vector by using the basis cycle embedding.

    Parameters
    ----------
    graph : nx.Graph or [nx.Graph]
        The graph or list of graphs.
    
    Returns
    -------
    np.ndarray
        The vector representation of the graph. If the input is a list of graphs, the output is a 2D array.
    """
    if isinstance(graph, list):
        return np.array([embedder.embed(g) for g in tqdm(graph, desc="Embedding graphs", total=len(graph))])
    return embedder.embed(graph)

def intracluster_distances(points, labels):
    """
    Compute the average centroid distance between points in the same cluster.

    Parameters
    ----------
    points : np.ndarray
        The points.
    labels : np.ndarray
        The labels of the points.

    Returns
    -------
    dict
        The intracluster distances.
    """
    clusters = np.unique(labels)
    intra_distances = {}
    for cluster in tqdm(clusters, desc="Computing intracluster distances", total=len(clusters)):
        members = points[labels == cluster]
        pairwise_distances = dist.pdist(members, 'euclidean')
        centroid = np.mean(members, axis=0)

        intra_distances[cluster] = {
            "pairwise": np.mean(pairwise_distances),
            "diameter": np.max(pairwise_distances),
            "centroid": np.mean(dist.cdist(members, [centroid], 'euclidean'))
        }
    return intra_distances

def intercluster_distances(points, labels):
    """
    Compute the average centroid distance different clusters.

    Parameters
    ----------
    points : np.ndarray
        The points.
    labels : np.ndarray
        The labels of the points.

    Returns
    -------
    dict
        The intercluster distances.
    """
    clusters = np.unique(labels)
    centroids = np.array([np.mean(points[labels == cluster], axis=0) for cluster in clusters])

    pairwise_distances = dist.pdist(centroids, 'euclidean')
    return {
        "avg": np.mean(pairwise_distances),
        "max": np.max(pairwise_distances),
        "min": np.min(pairwise_distances)
    }
    

if __name__ == "__main__":
    
    sample_size = 300
    np.random.seed(42)

    """
    Results:
    Average intra-cluster centroid distance: 3.518734118681113
    Average intra-cluster pairwise distance: 4.4082545804105955
    Average intra-cluster diameter: 9.427449813736837

    Average intercluster distance: 11.188578131182323
    Maximum intercluster distance: 62.03224967708329
    Minimum intercluster distance: 0.0
    """


    # Read file from data 
    files = os.listdir(os.path.join(THIS_FOLDER, "data/graphs"))
    
    graphs = []
    labels = []
    for file in tqdm(files, desc="Reading graphs", total=len(files)):
        path = os.path.join(THIS_FOLDER, "data/graphs", file)
        file_graphs = read_graphs(path)
        
        sample_idx = np.random.choice(len(file_graphs), sample_size, replace=False)
        sample = [file_graphs[i] for i in sample_idx]
        graphs += sample

        labels += [file] * sample_size

    # Transform graphs
    graphs = [modular_product(g, nx.path_graph(3)) for g in tqdm(graphs, desc="Transforming graphs", total=len(graphs))]

    # Convert graphs to vectors
    vectors = to_vec(graphs)
    labels = np.array(labels)

    # Compute intracluster distances
    intra_distances = intracluster_distances(vectors, labels)

    """
    for cluster, distances in intra_distances.items():
        print(f"Cluster {cluster}")
        print(f"Average centroid distance: {distances['centroid']}")
        print(f"Average pairwise distance: {distances['pairwise']}")
        print(f"Diameter: {distances['diameter']}")
        print()
    """
    print(f"Average intra-cluster centroid distance: {np.mean([distances['centroid'] for distances in intra_distances.values()])}")
    print(f"Average intra-cluster pairwise distance: {np.mean([distances['pairwise'] for distances in intra_distances.values()])}")
    print(f"Average intra-cluster diameter: {np.mean([distances['diameter'] for distances in intra_distances.values()])}")
    print()

    # Compute intercluster distances
    inter_distances = intercluster_distances(vectors, labels)

    print(f"Average intercluster distance: {inter_distances['avg']}")
    print(f"Maximum intercluster distance: {inter_distances['max']}")
    print(f"Minimum intercluster distance: {inter_distances['min']}")