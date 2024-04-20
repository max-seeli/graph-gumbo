from __future__ import annotations
import networkx as nx
from networkx import Graph
import numpy as np
from tqdm import tqdm

from typing import List, Union, Callable, Dict, Any, Hashable
from collections import defaultdict
from warnings import warn
  

class GraphEmbedding:

    def __call__(self, graph: Union[nx.Graph, List[nx.Graph]], verbose: bool = False):
        """
        Embed a graph into a vector.

        Parameters
        ----------
        graph : networkx.Graph, list of networkx.Graph
            The graph to embed.

        Returns
        -------
        embedding : numpy.ndarray
            The embedding of the graph.
        """
        if type(graph) is list:
            iter_graphs = tqdm(graph) if verbose else graph
            return np.array([self.__call__(g) for g in iter_graphs])
        return self.embed(graph)

    def embed(self, graph: nx.Graph):
        """
        Embed a graph into a vector.

        Parameters
        ----------
        graph : networkx.Graph
            The graph to embed.

        Returns
        -------
        embedding : numpy.ndarray
            The embedding of the graph.
        """
        raise NotImplementedError("The embed method must be implemented by a subclass.") 
    


class CountingEmbedding(GraphEmbedding):

    def __init__(self, count_function: Callable[[nx.Graph], Dict[int, int]], overflow_idx: Union[int, None] = None, size: int = None): 
        """
        Initialize the embedder with a counting function and an overflow index.

        Parameters
        ----------
        count_function : callable
            A function that takes a graph and returns a dictionary of counts.
        overflow_idx : int or None
            The index to use for counts that are greater than the size of the
            embedding. If None, a warning is issued when counts greater than
            the size of the embedding are found.
        """
        self.count_function = count_function
        self.overflow_idx = overflow_idx
        self.size = size


    def embed(self, graph: nx.Graph):
        """
        Embed a graph into a vector. The vector is of size `size`. The
        embedding is generated from the `self.count_function`.

        Parameters
        ----------
        graph : networkx.Graph
            The graph to embed.

        Returns
        -------
        embedding : numpy.ndarray
            The embedding of the graph.
        """
        counts = self.count_function(graph)
        return self._embed_counts(counts, self.size)

    def _embed_counts(self, counts, size, warn=True):
        """
        Embed the `counts` of a specific key (e.g. degrees or cycles in a graph)
        into a vector. The vector is of size `size`. All counts of indices
        greater than `size` are summed into the `self.overflow_idx`. After
        that, the index of the vector corresponds to the key of the counts.

        Parameters
        ----------
        counts : dict
            A dictionary mapping keys to the number of counts of that key.
        size : int
            The size of the embedding.
        warn : bool
            If True, warn when counts of cycles with length greater than `size` are
            found.

        Returns
        -------
        embedding : numpy.ndarray
            The embedding of the cycle counts.
        """
        embedding = np.zeros(size, dtype=np.int32)

        for key, count in counts.items():
            if key < size:
                embedding[key] = count
            elif self.overflow_idx is not None:
                embedding[self.overflow_idx] += count
            elif warn:
                    print("Warning: a count of key {} was found, but the embedding size is {}. Increase the embedding size to avoid this.".format(key, size))

        if self.overflow_idx is not None and warn and embedding[self.overflow_idx] > 0:
            print("Warning: {} counts of keys greater than {} were found. Increase the embedding size to avoid this.".format(embedding[self.overflow_idx], size))
        return embedding


class StructRule:

    def __init__(self, rule: Callable[[nx.Graph, List[int]], Hashable], categories: List[Hashable]):
        """
        Initialize the struct rule with a rule function and categories.

        Parameters
        ----------
        rule : callable
            A function that takes a graph and a struct and returns the
            appropriate category.
        categories : list
            The categories to use for the distinction.
        """
        self.rule = rule
        self.categories = categories
        self.category_idx = {cat: idx for idx, cat in enumerate(categories)}

    def __call__(self, graph: nx.Graph, struct: List[int]) -> Hashable:
        """
        Apply the rule to a graph and a structure.
        
        Parameters
        ----------
        graph : networkx.Graph
            The graph to apply the rule to.
        struct : list of int
            The structure to apply the rule to.
            
        Returns
        -------
        category : hashable
            The category of the structure.
        """
        return self.rule(graph, struct)
    
    def __len__(self) -> int:
        """
        Return the number of categories.

        Returns
        -------
        num_categories : int
            The number of categories.
        """
        return len(self.categories)
    
    def __add__(self, other: StructRule) -> StructRule:
        """
        Add two struct rules together. The new rule is a combination of the two
        rules and the new categories are the Cartesian product of the two
        categories.

        Parameters
        ----------
        other : StructRule
            The other struct rule to add.

        Returns
        -------
        new_rule : StructRule
            The new struct rule.
        """
        new_rule = lambda g, c: (self.rule(g, c), other.rule(g, c))
        new_categories = [(a, b) for a in self.categories for b in other.categories]
        return StructRule(new_rule, new_categories)

    def get_idx(self, category: Hashable) -> int:
        """
        Get the index of a category.

        Parameters
        ----------
        category : hashable
            The category to get the index of.
        
        Returns
        -------
        idx : int
            The index of the category.
        """

        return self.category_idx[category]
    

class StructureEmbedding(CountingEmbedding):

    def __init__(self, structure_iterator_function: Callable[[nx.Graph], List[Any]], rule: StructRule=None, **kwargs):
        """
        Initialize the structure embedder with a structure iterator function.
        
        Parameters
        ----------
        structure_iterator_function : callable
            A function that takes a graph and returns a list of structures.
        **kwargs : dict
            Additional keyword arguments to pass to the CountingEmbedding constructor.
        """
        super().__init__(self.count_structure_sizes, 0, **kwargs)
        self.structure_iterator_function = structure_iterator_function
        self.rule = rule

    def embed(self, graph: nx.Graph):
        """
        Embed a graph into a vector. The vector is of size `self.size`. The
        embedding is generated from the `self.count_function` over the
        structures of the graph given by the `self.structure_iterator_function`.
        If a `self.rule` is given, the embedding is generated from the categorized
        counts of the structures.

        Parameters
        ----------
        graph : networkx.Graph, list of networkx.Graph
            The graph to embed.

        Returns
        -------
        embedding : numpy.ndarray
            The embedding of the graph.
        """
        if self.rule is None:
            return super().embed(graph)
        else:
            categorized_counts = self.count_categorized_structure_sizes(graph, self.rule)
            category_size = self.size // len(self.rule)
            
            embedding = np.zeros(self.size, dtype=np.int32)
            for category, counts in categorized_counts.items():
                idx = self.rule.get_idx(category)
                embedding[idx*category_size:(idx+1)*category_size] = self._embed_counts(counts, category_size)
            return embedding

    def count_structure_sizes(self, graph: nx.Graph) -> Dict[int, int]:
        """
        Count the number of structures of each length for a graph.

        Parameters
        ----------
        graph : networkx.Graph
            The graph to count.
        
        Returns
        -------
        struct_counts : dict
            A dictionary mapping structure size to the number of structures of that size.
        """
        categorized_structure_counts = self.count_categorized_structure_sizes(
            graph, lambda g, s: "no_rule")
        return categorized_structure_counts["no_rule"]

    def count_categorized_structure_sizes(self, graph, rule):
        """
        Count the number of structures of each length for a graph, categorized by
        the given rule.

        Parameters
        ----------
        graph : networkx.Graph
            The graph to count.
        rule : StructRule
            The rule to use for categorization.

        Returns
        -------
        categorized_structure_counts : dict
            A dictionary mapping categories to dictionaries mapping structure size
            to the number of structures of that size.
        """

        categorized_structure_counts = defaultdict(dict)

        for struct in self.structure_iterator_function(graph):
            struct_size = len(struct)
            struct_category = rule(graph, struct) 
            struct_counts = categorized_structure_counts[struct_category]

            if struct_size not in struct_counts:
                struct_counts[struct_size] = 0
            struct_counts[struct_size] += 1
        return categorized_structure_counts

from basis_cycles import all_mst_basis_cycles, all_path_prod_cycle_basis
class BasisCycleEmbedding(StructureEmbedding):

    def __init__(self, **kwargs):
        """
        Initialize the basis cycle embedder.
        
        The basis cycle embedder counts the number of basis cycles in a graph.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the StructureEmbedding constructor.
        """
        super().__init__(all_path_prod_cycle_basis, **kwargs)


class ChordlessCycleEmbedding(StructureEmbedding):

    def __init__(self, **kwargs):
        """
        Initialize the chordless cycle embedder.

        The chordless cycle embedder counts the number of chordless cycles in a graph.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the StructureEmbedding constructor.
        """
        super().__init__(nx.chordless_cycles, **kwargs)

class KNeighborhoodEmbedding(StructureEmbedding):

    def __init__(self, k: int, multiset_neighbourhoods: bool = True, **kwargs):
        """
        Initialize the k-neighborhood embedder.

        The k-neighborhood embedder counts the size of k-neighborhoods in a graph.

        Parameters
        ----------
        k : int
            The size of the neighborhood.
        multiset_neighbourhoods : bool
            If True, the neighborhoods are multisets. If False, the neighborhoods
            are sets.
        **kwargs : dict
            Additional keyword arguments to pass to the StructureEmbedding constructor.
        """
        self.k = k
        self.multiset_neighbourhoods = multiset_neighbourhoods
        super().__init__(self.k_neighborhoods, **kwargs)

    def k_neighborhoods(self, graph: nx.Graph) -> List[List[int]]:
        """
        Get the k-neighborhoods of a graph.

        Parameters
        ----------
        graph : networkx.Graph
            The graph to get the k-neighborhoods of.

        Returns
        -------
        neighborhoods : list of list of int
            The k-neighborhoods of the graph.
        """
        neighborhoods = []
        for node in graph.nodes():
            neighborhood = [node]
            for _ in range(self.k):
                new_neighbors = []
                for neighbor in neighborhood:
                    new_neighbors.extend(graph.neighbors(neighbor))
                neighborhood.extend(new_neighbors)
            neighborhoods.append(neighborhood)

        if not self.multiset_neighbourhoods:
            neighborhoods = [list(set(neighborhood)) for neighborhood in neighborhoods]
        return neighborhoods

class TriangleEmbedding(StructureEmbedding):

    def __init__(self, **kwargs):
        """
        Initialize the triangle embedder.

        The triangle embedder counts the number of triangles in a graph.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the StructureEmbedding constructor.
        """
        super().__init__(self.all_triangles, **kwargs)

    def all_triangles(self, G):
        sorted_edges = []
        for edge in G.edges():
            sorted_edges.append(tuple(sorted(edge)))
        sorted_edges.sort()

        triangles = []
        for i in range(0, len(sorted_edges) - 1):
            for j in range(i + 1, len(sorted_edges)):
                if sorted_edges[i][0] == sorted_edges[j][0]:
                    if G.has_edge(sorted_edges[i][1], sorted_edges[j][1]):
                        triangles.append((sorted_edges[i][0], sorted_edges[i][1], sorted_edges[j][1]))
        return triangles

class DegreeEmbedding(CountingEmbedding):

    def __init__(self, **kwargs):
        """
        Initialize the degree embedder.
        
        The degree embedder counts the number of nodes with each specific node degree.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the CountingEmbedding constructor.
        """
        super().__init__(self.count_degrees, -1, **kwargs)

    def count_degrees(self, graph):
        """
        Count the number of nodes with each specific node degree.

        Parameters
        ----------
        graph : networkx.Graph
            The graph to count.

        Returns
        -------
        degree_counts : dict
            A dictionary mapping degrees to the number of nodes with the degree.
        """
        degree_counts = {}

        for node in graph.nodes():
            degree = graph.degree(node)
            if degree not in degree_counts:
                degree_counts[degree] = 0
            degree_counts[degree] += 1

        return degree_counts


class DegreeSequenceEmbedding(CountingEmbedding):

    def __init__(self, **kwargs):
        """
        Initialize the degree sequence embedder.

        The degree sequence embedder uses the degree sequence of a graph as the
        embedding.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the CountingEmbedding constructor.
        """
        super().__init__(self.count_degree_sequence, None, **kwargs)

    def count_degree_sequence(self, graph):
        """
        Given a graph, return a count-like representation of the degree sequence.

        Parameters
        ----------
        graph : networkx.Graph
            The graph to count.

        Returns
        -------
        degree_sequence_counts : dict
            A dictionary mapping positions in the degree sequence to the degree
            at that position
        """
        degree_sequence = sorted([d for _, d in graph.degree()], reverse=True)
        degree_sequence_counts = {k: v for k, v in enumerate(degree_sequence)}
        return degree_sequence_counts


class SpectralKAdjacencyEmbedding(GraphEmbedding):
    
        def __init__(self, k: int, size: int):
            """
            Initialize the spectral k-adjacency embedder.
    

    
            Parameters
            ----------
            k : int
                The number of eigenvalues to use.
            """
            self.k = k
            self.size = size
    
        def embed(self, graph: nx.Graph):
            eigen_vals =  self._embed_spectral_k_adjacency(graph, self.k)
            if self.size < len(eigen_vals):
                warn("The embedding size is smaller than the number of eigenvalues. Increase the embedding size to avoid this.")
                return eigen_vals[:self.size]
            else:
                # pad with nan value
                return np.pad(eigen_vals, (0, self.size - len(eigen_vals)), constant_values=np.nan)
            
        def _embed_spectral_k_adjacency(self, graph: nx.Graph, k: int):
            """
            Embed a graph into a vector using the eigenvalues of the k^th power of the
            adjacency matrix.
    
            Parameters
            ----------
            graph : networkx.Graph
                The graph to embed.
            k : int
                The number of eigenvalues to use.
    
            Returns
            -------
            embedding : numpy.ndarray
                The embedding of the graph.
            """
            adjacency_matrix = nx.to_numpy_array(graph)
            k_power_sum = np.zeros(adjacency_matrix.shape)
            for i in range(1, k+1):
                k_power_sum += np.linalg.matrix_power(adjacency_matrix, i)
            eigenvalues = np.linalg.eigvalsh(k_power_sum)
            return eigenvalues
        
from product.product_operator import rooted_product_permutation_family
class RootedProductFamilyEmbedding(GraphEmbedding):

    def __init__(self, factor: nx.Graph):
        """
        Initialize the rooted product family embedder.

        Parameters
        ----------
        factor : networkx.Graph
            The factor graph to use for generating the rooted forrest.
        """
        self.factor = factor

    def embed(self, graph: nx.Graph):
        
        return nx.weisfeiler_lehman_graph_hash(rooted_product_permutation_family(graph, self.factor))