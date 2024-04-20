"""
Train a model on the 'IMDB-BINARY' dataset from TU Dortmund.
Comparing the performance of the regular dataset as well as a transformed dataset (product graphs).
"""
import os
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, to_networkx, from_networkx
from torch_geometric.transforms import Compose, BaseTransform, OneHotDegree
from sklearn.model_selection import train_test_split


from model import GraphGIN
from trainer import PerformanceMetric, GNNTrainer

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoints')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class GenerateRootedForrest(BaseTransform):

    def __init__(self, factor):
        """
        Initialize the transform with a factor graph.

        Parameters
        ----------
        factor : nx.Graph
            The factor graph to use for generating the rooted forrest.
        """
        from product.product_operator import rooted_product_permutation_family
        self.rppf = rooted_product_permutation_family
        self.factor = factor

    def forward(self, data):
        """
        Generate the rooted forrest for the given graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The graph to generate the rooted forrest for.

        Returns
        -------
        torch_geometric.data.Data
            The rooted forrest of the given graph.
        """
        target = data.y
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)
        G = self.rppf(G, self.factor)
        data = from_networkx(G)
        data.y = target
        return data
    
class GenerateRamdomRooted(BaseTransform):

    def __init__(self, factor):
        """
        Initialize the transform with a factor graph.

        Parameters
        ----------
        factor : nx.Graph
            The factor graph to use for generating the rooted forrest.
        """
        self.factor = factor

    def forward(self, data):
        """
        Generate a random rooted product for the given graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The graph to generate the rooted forrest for.

        Returns
        -------
        torch_geometric.data.Data
            The rooted forrest of the given graph.
        """
        target = data.y
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)
        g_root = torch.randint(0, G.number_of_nodes(), (1,)).item()
        G = nx.rooted_product(self.factor, G, root=g_root)
        data = from_networkx(G)
        data.y = target
        return data

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
        self.is_rooted = args.rooted
        self.dataset = self.get_dataset(self.is_rooted)
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
        self.checkpoint_dir = CHECKPOINT_PATH
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
        Use rooted      = {self.is_rooted}
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
        Checkpoint Path = {self.checkpoint_dir}
        '''
        
    def get_dataset(self, is_rooted):
        """
        Get the dataset with the given transformation.

        Parameters
        ----------
        is_rooted : bool
            Whether to use the rooted product graph transformation.

        Returns
        -------
        torch_geometric.datasets.TUDataset
            The dataset with the given transformation.
        """
        if is_rooted:
            factor = nx.path_graph(2)
            transform = Compose([GenerateRootedForrest(factor), OneHotDegree(136)])
        else:
            transform = OneHotDegree(135)

        prefix = 'rooted' if is_rooted else 'normal'    
        path = os.path.join(DATA_PATH, prefix)
        os.makedirs(path, exist_ok=True)

        return TUDataset(path, name='IMDB-BINARY', pre_transform=transform)

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
    

    def get_trainer(self, model):
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

        train_metrics = PerformanceMetric(list(range(self.dataset.num_classes)))
        val_metrics = PerformanceMetric(list(range(self.dataset.num_classes)))

        train_dataset, test_dataset = self.get_splits()
        return GNNTrainer(model,
                          optimizer,
                          scheduler,
                          criterion,
                          self.epochs,
                          self.eval_every,
                          train_metrics,
                          val_metrics,
                          train_dataset,
                          test_dataset,
                          self.batch_size,
                          self.checkpoint_dir,
                          device)
   
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

    model = config.get_model()
    trainer = config.get_trainer(model)
    trainer.train()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train a GNN on the IMDB-BINARY dataset. (Optionally using rooted product graphs)')
    parser.add_argument('--rooted', action='store_true', help='Use rooted product graphs')
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
    args = parser.parse_args()

    config = ExperimentConfig(args)
    run(config)



