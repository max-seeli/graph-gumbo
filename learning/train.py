import os
from enum import Enum
from functools import partial
from warnings import warn

import networkx as nx
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from model import GraphGIN
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose, NormalizeFeatures, OneHotDegree
from trainer import GNNTester, GNNTrainer, PerformanceMetric
from transforms import BasisCycleTransform
from wandb_logger import WandBLogger

from utils import prepend_dict

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
CHECKPOINT_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'checkpoints')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Datasets(Enum):
    """
    An enumeration of the different datasets that can be used in the experiment.
    """
    IMDB_BINARY = {
        'name': 'IMDB-BINARY',
        'max_degree': 135
    }
    IMDB_MULTI = {
        'name': 'IMDB-MULTI',
        'max_degree': 88
    }
    REDDIT_BINARY = {
        'name': 'REDDIT-BINARY',
        'max_degree': 3062
    }
    SYNTHETIC = {
        'name': 'SYNTHETIC',
        'max_degree': 8
    }
    SYNTHIE = {
        'name': 'Synthie',
        'max_degree': 20
    }
    MUTAG = {
        'name': 'MUTAG',
        'max_degree': 4
    }
    MSRC_9 = {
        'name': 'MSRC_9',
        'max_degree': 16
    }
    ENZYMES = {
        'name': 'ENZYMES',
        'max_degree': 9
    }


class Transforms(Enum):
    """
    An enumeration of the different transformations that can be applied to the dataset.
    """
    DEGREE = 0
    BASIS_CYCLE_DEGREE = 1


class GraphProducts(Enum):
    """
    An enumeration of the different graph products that can be used in the product graph transformation.
    """
    CARTESIAN = partial(nx.cartesian_product)
    TENSOR = partial(nx.tensor_product)
    STRONG = partial(nx.strong_product)
    MODULAR = partial(nx.modular_product)


class FactorGraphs(Enum):
    """
    An enumeration of the different factor graphs that can be used in the product graph transformation.
    """
    PATH = partial(nx.path_graph)
    COMPLETE = partial(nx.complete_graph)
    STAR = partial(nx.star_graph)


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
        self.graph_product = args.graph_product
        self.factor_graph = args.factor_graph
        self.factor_size = args.factor_size
        self.embedding_size = args.embedding_size
        self.checkpoint_dir = CHECKPOINT_PATH

        self.dataset = self.get_dataset(args.dataset)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @property
    def run_name(self):
        """
        Get the name of the run based on the configuration.

        Returns
        -------
        str
            The name of the run based on the configuration.
        """
        if Transforms[self.transform] == Transforms.DEGREE:
            return f'{self.transform}'
        elif Transforms[self.transform] == Transforms.BASIS_CYCLE_DEGREE:
            return f'{self.transform}_{self.graph_product}_{self.factor_graph}_{self.factor_size}'

    def __repr__(self):
        """
        Get a string representation of the configuration.

        Returns
        -------
        str
            A string representation of the configuration.
        """
        dataset_balance = self.dataset._data.y.unique(return_counts=True)
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
        Graph Product   = {self.graph_product}
        Factor Graph    = {self.factor_graph}
        Factor Size     = {self.factor_size}
        Checkpoint Path = {self.checkpoint_dir}
        '''

    def get_dataset(self, dataset):
        """
        Get the dataset with the given transformation.

        Returns
        -------
        torch_geometric.datasets.TUDataset
            The dataset with the given transformation.
        """
        dataset = Datasets[dataset].value

        if Transforms[self.transform] == Transforms.DEGREE:
            transform = Compose([
                OneHotDegree(max_degree=dataset['max_degree']),
                T.AddSelfLoops()])
            desc = self.transform
        elif Transforms[self.transform] == Transforms.BASIS_CYCLE_DEGREE:
            fg = FactorGraphs[self.factor_graph].value(self.factor_size)
            gp = GraphProducts[self.graph_product].value
            transform = Compose([
                BasisCycleTransform(fg, gp, emb_size=self.embedding_size),
                NormalizeFeatures(),
                OneHotDegree(max_degree=dataset['max_degree']),
                T.AddSelfLoops()])
            desc = f'{self.transform}-{self.graph_product}-{self.factor_graph}-{self.factor_size}'

        dataset_path = os.path.join(DATA_PATH, f'{dataset["name"]}-{desc}')
        os.makedirs(dataset_path, exist_ok=True)
        return TUDataset(dataset_path, name=dataset['name'], pre_transform=transform, use_node_attr=True)

    def get_splits(self):
        """
        Get the training and test splits for the experiment.

        Returns
        -------
        tuple
            The training and test splits for the experiment.
        """
        dataset = self.dataset.shuffle()
        train, test = train_test_split(
            range(len(dataset)), train_size=self.train_size, stratify=dataset.y)
        return dataset[train], dataset[test]

    def get_model(self, checkpoint_path=None):
        """
        Get the model to use for the experiment.

        Parameters
        ----------
        checkpoint_path : optional, str
            The path to a checkpoint to load the model from.

        Returns
        -------
        GraphGIN
            The model to use for the experiment.
        """
        model = GraphGIN(in_channels=self.dataset.num_features,
                         hidden_channels=self.hidden_channels,
                         out_channels=self.dataset.num_classes,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         act="relu",
                         act_first=False,
                         norm='batch',
                         jk="cat").to(device)

        if checkpoint_path is not None:
            model.load_state_dict(torch.load(checkpoint_path))
        return model

    def get_trainer(self, model: GraphGIN, train_dataset=None, val_dataset=None) -> GNNTrainer:
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
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.patience, gamma=self.gamma)
        criterion = torch.nn.CrossEntropyLoss()

        train_metrics = PerformanceMetric(
            list(range(self.dataset.num_classes)), device=device)
        val_metrics = PerformanceMetric(
            list(range(self.dataset.num_classes)), device=device)

        if train_dataset is None and val_dataset is None:
            train_dataset, val_dataset = self.get_splits()

        wandb_logger = WandBLogger(
            enabled=True, model=model, run_name=self.run_name, notes=self.__repr__())

        return GNNTrainer(model,
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
                          device,
                          wandb_logger)

    def get_tester(self, model: GraphGIN, test_dataset) -> GNNTester:
        """
        Get the tester to use for the experiment.

        Parameters
        ----------
        model : GraphGIN
            The model to use for the experiment.

        Returns
        -------
        GNNTester
            The tester to use for the experiment.
        """
        criterion = torch.nn.CrossEntropyLoss()
        test_metrics = PerformanceMetric(
            list(range(self.dataset.num_classes)), device=device)
        return GNNTester(model, test_metrics, test_dataset, criterion, self.batch_size, device)


class Experiment:

    def __init__(self, config: ExperimentConfig):
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
                trainer = self.config.get_trainer(model, train, val)
                run_summary = trainer.train(fold)

                best_model = self.config.get_model(
                    trainer.get_checkpoint_path(trainer.best_val_epoch, fold))
                tester = self.config.get_tester(best_model, test_dataset)
                test_summary = tester.test()
                test_summary = prepend_dict(test_summary, 'test_')
                print(test_summary)

                fold_metrics.append({
                    **run_summary,
                    **test_summary})

            print(f'Average metrics over {self.config.folds} folds:')
            avg_metrics = {k: sum(m[k] for m in fold_metrics) /
                           len(fold_metrics) for k in fold_metrics[0].keys()}
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

    Experiment(config).run()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='Train a GNN on the IMDB-BINARY dataset.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The number of epochs to train for')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='The number of hidden channels to use in the GNN')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='The number of layers to use in the GNN')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='The dropout rate to use in the GNN')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The learning rate to use for training')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='The gamma value to use for the learning rate scheduler')
    parser.add_argument('--patience', type=int, default=50,
                        help='The number of epochs to wait before reducing the learning rate')
    parser.add_argument('--eval_every', type=int, default=2,
                        help='The number of epochs to wait before evaluating the model')
    parser.add_argument('--train_size', type=float, default=0.8,
                        help='The number of training graphs to use')
    parser.add_argument('--folds', type=int, default=10,
                        help='The number of folds to use for cross-validation')
    parser.add_argument('--dataset', type=str, default='IMDB_BINARY', choices=[
                        d.name for d in Datasets], help='The dataset to use for the experiment')
    parser.add_argument('--transform', type=str, default='DEGREE', choices=[
                        t.name for t in Transforms], help='The transformation to apply to the dataset')

    parser.add_argument('--graph_product', type=str, default="CARTESIAN", choices=[
                        p.name for p in GraphProducts], help='The graph product to use for the product graph transformation')
    parser.add_argument('--factor_graph', type=str, default='PATH', choices=[
                        f.name for f in FactorGraphs], help='The factor graph to use in case of the product graph transformation')
    parser.add_argument('--factor_size', type=int, default=5,
                        help='The size of the factor graph to use in case of the product graph transformation')
    parser.add_argument('--embedding_size', type=int, default=10,
                        help='The size of the embedding to use in case of the basis cycle transformation')
    args = parser.parse_args()

    config = ExperimentConfig(args)
    run(config)
