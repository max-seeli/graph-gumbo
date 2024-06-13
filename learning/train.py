"""
Train a model on the 'IMDB-BINARY' dataset from TU Dortmund.
Comparing the performance of the regular dataset as well as a transformed dataset (product graphs).
"""
import os
from enum import Enum
import torch
import torch.nn.functional as F
import networkx as nx
from warnings import warn
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, to_networkx, from_networkx
from torch_geometric.transforms import Compose, BaseTransform, OneHotDegree, NormalizeFeatures
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split, StratifiedKFold

from product.product_operator import modular_product
from counting import BasisCycleEmbedding
from model import GraphGIN
from trainer import PerformanceMetric, GNNTrainer

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoints')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class BasisCycleTransform(BaseTransform):

    def __init__(self, factor=None, emb_size=20, cat=True):
        """
        Initialize the transform with a factor graph.

        Parameters
        ----------
        factor : optional, networkx.Graph
            The factor graph to use before computing the basis cycle embedding.
        """
        self.factor = factor
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
        #bce = torch.tensor(self.embedder(G), dtype=torch.float) # (emb_size)
        
        #self.per_node_basis_cycle(G)
        # bce_rep = bce.unsqueeze(0).repeat(data.num_nodes, 1)
        
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
            G_prod = modular_product(G, self.factor)
            idx = lambda node: node[0]
        else:
            G_prod = G
            idx = lambda node: node

        for cycle in nx.cycle_basis(G_prod):
            cycle_length = len(cycle)
            if cycle_length > self.emb_size + 2:
                cycle_length = self.emb_size + 2
                warn(f'Cycle length {len(cycle)} is greater than the embedding size {self.emb_size}. Truncating the cycle to {self.emb_size}.')

            full_embedding[cycle_length - 3] += 1
            for node in cycle:
                per_node_embedding[idx(node), cycle_length - 3] += 1

        return per_node_embedding, full_embedding
       

class Transforms(Enum):
    """
    An enumeration of the different transformations that can be applied to the dataset.
    """
    DEGREE = Compose([
            OneHotDegree(max_degree=135),
            T.AddSelfLoops()])
    BASIS_CYCLE_DEGREE = Compose([
        BasisCycleTransform(nx.path_graph(5), emb_size=10), 
        NormalizeFeatures(),
        OneHotDegree(max_degree=136),
        T.AddSelfLoops()])
    

class ExperimentConfig:
    """
    A class to store the configuration of an experiment.
    """

    def __init__(self, args):
        """
        Initialize the configuration with the given transformation.

        Parameters
        ----------
        args : argparse.Namespace
            The arguments to use for the configuration.
        """
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.hidden_channels = args.hidden_channels
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.lr = args.lr
        self.gamma = args.gamma
        self.patience = args.patience
        self.eval_every = args.eval_every
        self.train_size = args.train_size
        self.folds = args.folds
        self.transform = args.transform
        self.checkpoint_dir = CHECKPOINT_PATH

        self.dataset = self.get_dataset()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def __repr__(self):
        """
        Get a string representation of the configuration.

        Returns
        -------
        str
            A string representation of the configuration.
        """
        dataset_balance = self.dataset.data.y.unique(return_counts=True)
        return f'''
        The configuration is as follows:
        Dataset         = {self.dataset}
        Dataset balance = {dict(zip(dataset_balance[0].tolist(), dataset_balance[1].tolist()))}
        Node Features   = {self.dataset.num_features}
        Batch size      = {self.batch_size}
        Epochs          = {self.epochs}
        Hidden Channels = {self.hidden_channels}
        Num Layers      = {self.num_layers}
        Dropout         = {self.dropout}
        Learning Rate   = {self.lr}
        Gamma           = {self.gamma}
        Patience        = {self.patience}
        Eval Every      = {self.eval_every}
        Train Size      = {self.train_size}
        Folds           = {self.folds}
        Transform       = {self.transform}
        Checkpoint Path = {self.checkpoint_dir}
        '''
        
    def get_dataset(self):
        """
        Get the dataset with the given transformation.

        Returns
        -------
        torch_geometric.datasets.TUDataset
            The dataset with the given transformation.
        """
        os.system(f'rm -r {DATA_PATH}')
        transform = Transforms[self.transform].value

        os.makedirs(DATA_PATH, exist_ok=True)
        return TUDataset(DATA_PATH, name='IMDB-BINARY', pre_transform=transform)

    def get_splits(self):
        """
        Get the training and test splits for the experiment.

        Returns
        -------
        tuple
            The training and test splits for the experiment.
        """
        dataset = self.dataset.shuffle()
        train, test = train_test_split(range(len(dataset)), train_size=self.train_size/100, stratify=dataset.y)
        return dataset[train], dataset[test]

    def get_model(self):
        """
        Get the model to use for the experiment.

        Returns
        -------
        GraphGIN
            The model to use for the experiment.
        """
        return GraphGIN(in_channels=self.dataset.num_features,
                        hidden_channels=self.hidden_channels,
                        out_channels=self.dataset.num_classes,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        act="relu",
                        act_first=False,
                        norm='batch',
                        jk="cat").to(device)
    

    def get_trainer(self, model, run_name=None, train_dataset=None, val_dataset=None):
        """
        Get the trainer to use for the experiment.

        Parameters
        ----------
        model : GraphGIN
            The model to use for the experiment.

        Returns
        -------
        GNNTrainer
            The trainer to use for the experiment.
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.patience, gamma=self.gamma)
        criterion = torch.nn.CrossEntropyLoss()

        train_metrics = PerformanceMetric(list(range(self.dataset.num_classes)), device=device)
        val_metrics = PerformanceMetric(list(range(self.dataset.num_classes)), device=device)

        if train_dataset is None and val_dataset is None:
            train_dataset, val_dataset = self.get_splits()
        return GNNTrainer(run_name,
                          model,
                          optimizer,
                          scheduler,
                          criterion,
                          self.epochs,
                          self.eval_every,
                          train_metrics,
                          val_metrics,
                          train_dataset,
                          val_dataset,
                          self.batch_size,
                          self.checkpoint_dir,
                          device)
    
class Experiment:

    def __init__(self, config):
        """
        Initialize the experiment with the given configuration.

        Parameters
        ----------
        config : ExperimentConfig
            The configuration to use for the experiment.
        """
        self.config = config

    def run(self):
        """
        Run the experiment with the given configuration.
        """
        torch.manual_seed(42)
        print(self.config)

        if self.config.folds > 1:
            train_dataset, test_dataset = self.config.get_splits()
            skf = StratifiedKFold(n_splits=self.config.folds, shuffle=True)
            
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(train_dataset, train_dataset.y)):
                print(f'Fold {fold + 1}/{self.config.folds}')
                model = self.config.get_model()
                train = train_dataset[train_idx]
                val = train_dataset[val_idx]
                run_name = f'{self.config.transform}_{fold}'
                trainer = self.config.get_trainer(model, run_name, train, val)
                run_summary = trainer.train()
                fold_metrics.append(run_summary)

            print(f'Average metrics over {self.config.folds} folds:')
            avg_metrics = {k: sum(m[k] for m in fold_metrics) / len(fold_metrics) for k in fold_metrics[0].keys()}
            for k, v in avg_metrics.items():
                print(f'{k}: {v}')
        else: 
            model = self.config.get_model()

            trainer = self.config.get_trainer(model)
            print(trainer.train())

    def __repr__(self):
        """
        Get a string representation of the experiment.

        Returns
        -------
        str
            A string representation of the experiment.
        """
        return f'Experiment with configuration: {self.config}'
   
def run(config):
    """
    Run an experiment with the given configuration.

    Parameters
    ----------
    config : ExperimentConfig
        The configuration to use for the experiment.
    """
    torch.manual_seed(42)
    
    print(config)

    Experiment(config).run()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train a GNN on the IMDB-BINARY dataset.')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size to use for training')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train for')
    parser.add_argument('--hidden_channels', type=int, default=256, help='The number of hidden channels to use in the GNN')
    parser.add_argument('--num_layers', type=int, default=3, help='The number of layers to use in the GNN')
    parser.add_argument('--dropout', type=float, default=0.4, help='The dropout rate to use in the GNN')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate to use for training')
    parser.add_argument('--gamma', type=float, default=0.5, help='The gamma value to use for the learning rate scheduler')
    parser.add_argument('--patience', type=int, default=50, help='The number of epochs to wait before reducing the learning rate')
    parser.add_argument('--eval_every', type=int, default=2, help='The number of epochs to wait before evaluating the model')
    parser.add_argument('--train_size', type=int, default=80, help='The number of training graphs to use (in percentage)')
    parser.add_argument('--folds', type=int, default=1, help='The number of folds to use for cross-validation')
    parser.add_argument('--transform', type=str, default='DEGREE', choices=[
        'DEGREE',
        'BASIS_CYCLE_DEGREE'
    ], help='The transformation to apply to the dataset')
    args = parser.parse_args()

    config = ExperimentConfig(args)
    run(config)



