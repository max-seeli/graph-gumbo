import os
import time
from itertools import product

from tqdm import tqdm
from train import Datasets, FactorGraphs, GraphProducts, Transforms

epochs = 200
folds = 10
train_size = 0.8  # 80% of the data is used for training and 20% for validation

datasets = [d.name for d in Datasets]

configurations = [
    {
        'transform': Transforms.DEGREE.name
    },
    {
        'transform': Transforms.BASIS_CYCLE_DEGREE.name,
        'graph_product': GraphProducts.CARTESIAN.name,
        'factor_graph': FactorGraphs.COMPLETE.name,
        'factor_size': 3,
        'embedding_size': 75
    },
    {
        'transform': Transforms.BASIS_CYCLE_DEGREE.name,
        'graph_product': GraphProducts.MODULAR.name,
        'factor_graph': FactorGraphs.PATH.name,
        'factor_size': 3
    }
]

hyperparameters = {
    'batch_size': [32, 256],
    'hidden_channels': [64, 128],
    'dropout': [0.2],
    'lr': [0.001],
}

hyper_configs = product(*hyperparameters.values())
for j, hparams in enumerate(tqdm(list(hyper_configs))):
    print(f'{j}: {hparams}')

train_file = os.path.join(os.path.dirname(__file__), 'train.py')
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

for dataset in datasets:
    print(f'Running experiments for dataset {dataset}')
    print(f'Starting at {time.ctime()}')

    for i, config in enumerate(configurations):
        print(f'\tRunning experiments for configuration {config} on index {i}')
        print(f'\tStarting at {time.ctime()}')

        hyper_configs = product(*hyperparameters.values())
        for j, hparams in enumerate(tqdm(list(hyper_configs))):
            print(
                f'\n\t\tRunning experiments for hyperparameters {hparams} on index {j}')
            print(f'\t\tStarting at {time.ctime()}')

            batch_size, hidden_channels, dropout, lr = hparams

            config_str = ' '.join([f'--{k} {v}' for k, v in config.items()])

            command = f'python {train_file} --epochs {epochs} --folds {folds} --train_size {train_size} --dataset {dataset} --batch_size {batch_size} --hidden_channels {hidden_channels} --dropout {dropout} --lr {lr} {config_str}'
            # run command and redirect all output and error to log file
            os.system(f'{command} > {log_dir}/{dataset}_{i}_{j}.log 2>&1')

            print(f'\t\tFinished at {time.ctime()}')
