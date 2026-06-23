# decart/experiments/datasets/__init__.py
"""DeCart experiment datasets."""

from .mnist import MNISTDataLoader, load_mnist, load_mnist_records, visualize_mnist_sample
from .uci_har import UCIHARDataLoader, load_uci_har, load_uci_har_records


DATASET_SPECS = {
    'mnist': {
        'input_dim': 784,
        'num_classes': 10,
    },
    'uci_har': {
        'input_dim': 561,
        'num_classes': 6,
    },
}


def get_dataset_spec(dataset_name: str):
    return DATASET_SPECS.get(dataset_name)


def load_experiment_records(dataset_name: str, num_records: int, data_dir: str = './data', split: str = 'test'):
    if dataset_name == 'mnist':
        return load_mnist_records(num_records=num_records, data_dir=data_dir, split=split)
    if dataset_name == 'uci_har':
        return load_uci_har_records(num_records=num_records, data_dir=data_dir, split=split)
    raise ValueError(f'Unsupported dataset: {dataset_name}')

__all__ = [
    'MNISTDataLoader',
    'load_mnist',
    'load_mnist_records',
    'UCIHARDataLoader',
    'load_uci_har',
    'load_uci_har_records',
    'get_dataset_spec',
    'load_experiment_records',
    'visualize_mnist_sample'
]

__version__ = "1.0.0"
__description__ = "DeCart 实验数据集库 - MNIST / UCI HAR"
