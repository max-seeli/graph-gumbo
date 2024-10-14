from warnings import warn

import networkx as nx
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx


class BasisCycleTransform(BaseTransform):

    def __init__(self, factor=None, graph_product=None, emb_size=20, cat=True):
        """
        Initialize the transform with a factor graph.

        Parameters
        ----------
        factor : optional, networkx.Graph
            The factor graph to use before computing the basis cycle embedding.
        graph_product : optional, callable
            The graph product to use before computing the basis cycle embedding.
        emb_size : int
            The size of the basis cycle embedding.
        cat : bool
            Whether to concatenate the basis cycle embedding to the node features
        """
        self.factor = factor
        self.graph_product = graph_product
        self.emb_size = emb_size
        self.cat = cat

    def forward(self, data):
        """
        Generate the basis cycle embedding for the given graph and append it to the node features.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The graph to compute the basis cycle embedding for.

        Returns
        -------
        torch_geometric.data.Data
            The graph with the basis cycle embedding appended to the node features.
        """
        target = data.y
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)

        per_node_embedding, full_embedding = self.per_node_basis_cycle(G)

        # if self.factor is not None:
        #     G = modular_product(G, self.factor)
        #     data = from_networkx(G)
        # bce = torch.tensor(self.embedder(G), dtype=torch.float) # (emb_size)

        # self.per_node_basis_cycle(G)
        # bce_rep = bce.unsqueeze(0).repeat(data.num_nodes, 1)

        # normalize the basis cycle embedding
        per_node_embedding /= per_node_embedding.sum(dim=-1, keepdim=True)

        if data.x is not None and self.cat:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, per_node_embedding.to(x.dtype)], dim=-1)
        else:
            data.x = per_node_embedding

        data.y = target
        return data

    def per_node_basis_cycle(self, G):
        """
        Compute the basis cycle embedding for each node in the given graph.

        A basis cycle is connected to a node if the node is part of the cycle.

        Parameters
        ----------
        G : networkx.Graph
            The graph to compute the basis cycle embedding for.

        Returns
        -------
        torch.Tensor
            The basis cycle embedding for each node in the graph.
        """
        per_node_embedding = torch.zeros(len(G.nodes), self.emb_size)
        full_embedding = torch.zeros(self.emb_size)

        if self.factor is not None:
            G_prod = self.graph_product(G, self.factor)
            def idx(node): return node[0]
        else:
            G_prod = G
            def idx(node): return node

        for cycle in nx.cycle_basis(G_prod):
            cycle_length = len(cycle)
            if cycle_length > self.emb_size + 2:
                cycle_length = self.emb_size + 2
                warn(
                    f'Cycle length {len(cycle)} is greater than the embedding size {self.emb_size}. Truncating the cycle to {self.emb_size}.')

            full_embedding[cycle_length - 3] += 1
            for node in cycle:
                per_node_embedding[idx(node), cycle_length - 3] += 1

        return per_node_embedding, full_embedding
