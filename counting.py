from __future__ import annotations
import networkx as nx
import numpy as np

from typing import List, Union, Callable, Dict, Any, Hashable
from collections import defaultdict
  

class GraphEmbedding:

    def __init__(self, count_function: Callable[[nx.Graph], Dict[int, int]], overflow_idx: Union[int, None] = None): 
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

    def __call__(self, graph: Union[nx.Graph, List[nx.Graph]], size: int):
        """
        Embed a `graph` into a vector. The vector is of size `size`. The
        embedding is generated from the `self.count_function`.

        Parameters
        ----------
        graph : networkx.Graph, list of networkx.Graph
            The graph to embed.
        size : int
            The size of the embedding.

        Returns
        -------
        embedding : numpy.ndarray
            The embedding of the graph.
        """
        if type(graph) is list:
            return np.array([self.__call__(g, size) for g in graph])

        counts = self.count_function(graph)
        return self._embed_counts(counts, size)


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
    

class StructureEmbedding(GraphEmbedding):

    def __init__(self, structure_iterator_function: Callable[[nx.Graph], List[Any]]):
        """
        Initialize the structure embedder with a structure iterator function.
        
        Parameters
        ----------
        structure_iterator_function : callable
            A function that takes a graph and returns a list of structures.
        """
        super().__init__(self.count_structure_sizes, 0)
        self.structure_iterator_function = structure_iterator_function

    def __call__(self, 
                 graph: Union[nx.Graph, List[nx.Graph]],
                 size: int,
                 rule: StructRule = None) -> np.ndarray:
        """
        Embed a graph into a vector. The vector is of size `size`. The
        embedding is generated from the `self.count_function` over the
        structures of the graph given by the `self.structure_iterator_function`.
        If a rule is given, the embedding is generated from the categorized
        counts of the structures.

        Parameters
        ----------
        graph : networkx.Graph, list of networkx.Graph
            The graph to embed.
        size : int
            The size of the embedding.
        rule : StructRule or None
            The rule to use for the embedding. If None, no rule is used.

        Returns
        -------
        embedding : numpy.ndarray
            The embedding of the graph.
        """
        if type(graph) is list:
            return np.array([self.__call__(g, size, rule) for g in graph])

        if rule is None:
            return super().__call__(graph, size)
        else:
            categorized_counts = self.count_categorized_structure_sizes(graph, rule)
            category_size = size // len(rule)
            
            embedding = np.zeros(size, dtype=np.int32)
            for category, counts in categorized_counts.items():
                idx = rule.get_idx(category)
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

class BasisCycleEmbedding(StructureEmbedding):

    def __init__(self):
        """
        Initialize the basis cycle embedder.
        
        The basis cycle embedder counts the number of basis cycles in a graph.
        """
        super().__init__(nx.cycle_basis)


class ChordlessCycleEmbedding(StructureEmbedding):

    def __init__(self):
        """
        Initialize the chordless cycle embedder.

        The chordless cycle embedder counts the number of chordless cycles in a graph.
        """
        super().__init__(nx.chordless_cycles)

class DegreeEmbedding(GraphEmbedding):

    def __init__(self):
        """
        Initialize the degree embedder.
        
        The degree embedder counts the number of nodes with each specific node degree.
        """
        super().__init__(self.count_degrees, -1)

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


class DegreeSequenceEmbedding(GraphEmbedding):

    def __init__(self):
        """
        Initialize the degree sequence embedder.

        The degree sequence embedder uses the degree sequence of a graph as the
        embedding.
        """
        super().__init__(self.count_degree_sequence, None)

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
