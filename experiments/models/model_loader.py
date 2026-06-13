"""Helpers for loading trained experiment models from pickle artifacts."""

import glob
import os
import pickle
from typing import Any, Dict, Iterable, List


MODEL_PATTERNS = {
    'dot': ['dot_*.pkl'],
    'decision_tree': ['decision_tree_*.pkl'],
    'neural_network': ['cnn_flattened_*.pkl', 'mlp_*.pkl'],
}

DATASET_INPUT_DIMS = {
    'mnist': 784,
    'uci_har': 561,
}


def _candidate_paths(models_dir: str, patterns: Iterable[str]) -> List[str]:
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(models_dir, pattern)))
    return sorted(candidates, key=os.path.getmtime, reverse=True)


def _matches_dataset(config: Dict[str, Any], model_path: str, dataset_name: str | None) -> bool:
    if dataset_name is None:
        return True

    config_dataset = config.get('dataset_name') or config.get('architecture', {}).get('dataset_name')
    if config_dataset:
        return config_dataset == dataset_name

    basename = os.path.basename(model_path).lower()
    if dataset_name in basename:
        return True

    expected_input_dim = DATASET_INPUT_DIMS.get(dataset_name)
    architecture = config.get('architecture', {})
    input_dim = architecture.get('input_dim')
    if input_dim is not None and expected_input_dim is not None:
        return int(input_dim) == expected_input_dim

    return dataset_name == 'mnist'


def resolve_model_path(model_type: str, models_dir: str, dataset_name: str | None = None) -> str:
    patterns = MODEL_PATTERNS.get(model_type)
    if not patterns:
        raise ValueError(f'Unsupported model type: {model_type}')

    candidates = _candidate_paths(models_dir, patterns)
    if not candidates:
        raise FileNotFoundError(
            f'No trained model found for {model_type} under {models_dir}. '
            f'Run experiments/models/train_models.py first.'
        )

    for candidate in candidates:
        config = _load_pickle(candidate)
        if _matches_dataset(config, candidate, dataset_name):
            return candidate

    raise FileNotFoundError(
        f'No trained model found for {model_type} and dataset {dataset_name} under {models_dir}. '
        f'Run experiments/models/train_models.py --dataset {dataset_name} first.'
    )


def _load_pickle(model_path: str) -> Dict[str, Any]:
    with open(model_path, 'rb') as file_obj:
        return pickle.load(file_obj)


def _convert_neural_network(architecture: Dict[str, Any]) -> Dict[str, Any]:
    weights = architecture.get('combined_weights', architecture.get('weights', []))
    bias = architecture.get('combined_bias', architecture.get('bias', []))
    if not weights:
        raise ValueError('Trained neural-network model is missing weights')

    output_dim = int(architecture.get('output_dim', 10))
    input_dim = int(architecture.get('input_dim', 784))
    return {
        'type': 'neural_network',
        'format': 'single_layer',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'weights': list(weights),
        'bias': list(bias),
        'weights_shape': tuple(architecture.get('weights_shape', (output_dim, input_dim))),
        'bias_shape': tuple(architecture.get('bias_shape', (output_dim,))),
    }


def _convert_decision_tree(architecture: Dict[str, Any]) -> Dict[str, Any]:
    if architecture.get('type') != 'decision_tree' and 'nodes' not in architecture:
        raise ValueError('Trained decision-tree model is missing nodes')

    return {
        'type': 'decision_tree',
        'format': architecture.get('format', 'dict'),
        'root': architecture.get('root', 0),
        'nodes': list(architecture.get('nodes', [])),
    }


def _convert_dot(architecture: Dict[str, Any]) -> List[float]:
    weights = architecture.get('weights', [])
    if not weights:
        raise ValueError('Trained dot model is missing weights')
    return list(weights)


def load_trained_experiment_model(model_type: str, models_dir: str, dataset_name: str | None = None) -> Any:
    model_path = resolve_model_path(model_type, models_dir, dataset_name=dataset_name)
    config = _load_pickle(model_path)
    architecture = config.get('architecture', {})

    if model_type == 'neural_network':
        model = _convert_neural_network(architecture)
    elif model_type == 'decision_tree':
        model = _convert_decision_tree(architecture)
    elif model_type == 'dot':
        model = _convert_dot(architecture)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    print(f"   Loaded trained {model_type} model: {os.path.basename(model_path)}")
    return model
