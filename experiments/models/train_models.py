"""Train real experiment models for MNIST and UCI HAR."""

import argparse
import os
import sys
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from experiments.datasets import MNISTDataLoader, UCIHARDataLoader

MODEL_SAVE_DIR = os.path.join(project_dir, 'experiments', 'models', 'trained')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 45
LEARNING_RATE = 0.05
TREE_FEATURE_CANDIDATES = 64
TREE_THRESHOLD_QUANTILES = (0.2, 0.4, 0.6, 0.8)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    input_dim: int
    num_classes: int
    target_class: int
    batch_size: int
    epochs: int
    train_limit: int | None
    test_limit: int | None


DATASET_CONFIGS = {
    'mnist': DatasetConfig(
        name='mnist',
        input_dim=784,
        num_classes=10,
        target_class=0,
        batch_size=128,
        epochs=3,
        train_limit=12000,
        test_limit=2000,
    ),
    'uci_har': DatasetConfig(
        name='uci_har',
        input_dim=561,
        num_classes=6,
        target_class=0,
        batch_size=128,
        epochs=6,
        train_limit=None,
        test_limit=None,
    ),
}


class FlattenLinearClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.view(inputs.size(0), -1)
        return self.linear(flattened)


class DotBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.view(inputs.size(0), -1)
        return self.linear(flattened).squeeze(-1)


@dataclass
class DatasetBundle:
    config: DatasetConfig
    train_loader: DataLoader
    test_loader: DataLoader
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray



def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    actual = min(limit, len(dataset))
    if actual == len(dataset):
        return dataset
    return Subset(dataset, list(range(actual)))



def _bundle_from_mnist(config: DatasetConfig, data_dir: str) -> DatasetBundle:
    loader = MNISTDataLoader(data_dir=data_dir, batch_size=config.batch_size)
    loader.load_data(use_validation=False, download=True)

    train_dataset = _limit_dataset(loader.train_dataset, config.train_limit)
    test_dataset = _limit_dataset(loader.test_dataset, config.test_limit)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    numpy_data = loader.get_numpy_data(flatten=True)
    train_x, train_y = numpy_data['train']
    test_x, test_y = numpy_data['test']
    if config.train_limit is not None:
        train_x = train_x[:config.train_limit]
        train_y = train_y[:config.train_limit]
    if config.test_limit is not None:
        test_x = test_x[:config.test_limit]
        test_y = test_y[:config.test_limit]

    return DatasetBundle(config, train_loader, test_loader, train_x, train_y, test_x, test_y)



def _bundle_from_uci_har(config: DatasetConfig, data_dir: str) -> DatasetBundle:
    loader = UCIHARDataLoader(data_dir=data_dir, batch_size=config.batch_size)
    loader.load_data()

    train_dataset = _limit_dataset(loader.train_dataset, config.train_limit)
    test_dataset = _limit_dataset(loader.test_dataset, config.test_limit)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    numpy_data = loader.get_numpy_data()
    train_x, train_y = numpy_data['train']
    test_x, test_y = numpy_data['test']
    if config.train_limit is not None:
        train_x = train_x[:config.train_limit]
        train_y = train_y[:config.train_limit]
    if config.test_limit is not None:
        test_x = test_x[:config.test_limit]
        test_y = test_y[:config.test_limit]

    return DatasetBundle(config, train_loader, test_loader, train_x, train_y, test_x, test_y)



def build_dataset_bundle(dataset_name: str, data_dir: str) -> DatasetBundle:
    config = DATASET_CONFIGS[dataset_name]
    if dataset_name == 'mnist':
        return _bundle_from_mnist(config, data_dir)
    if dataset_name == 'uci_har':
        return _bundle_from_uci_har(config, data_dir)
    raise ValueError(f'Unsupported dataset: {dataset_name}')



def evaluate_multiclass(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(features)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
    return correct / total if total else 0.0



def evaluate_binary(model: nn.Module, data_loader: DataLoader, target_class: int) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(DEVICE)
            labels = (labels == target_class).float().to(DEVICE)
            logits = model(features)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
    return correct / total if total else 0.0



def train_neural_network(bundle: DatasetBundle) -> Tuple[FlattenLinearClassifier, Dict[str, List[float]], float]:
    print('\n' + '=' * 60)
    print(f'训练单层线性分类器 ({bundle.config.name} {bundle.config.input_dim} -> {bundle.config.num_classes})')
    print('=' * 60)

    model = FlattenLinearClassifier(bundle.config.input_dim, bundle.config.num_classes).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(bundle.config.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for features, labels in bundle.train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_total += int(labels.size(0))

        train_loss = running_loss / running_total if running_total else 0.0
        train_acc = running_correct / running_total if running_total else 0.0
        test_acc = evaluate_multiclass(model, bundle.test_loader)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(test_acc)
        print(f'   epoch {epoch + 1}/{bundle.config.epochs}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}')

    final_acc = evaluate_multiclass(model, bundle.test_loader)
    return model, history, final_acc



def train_dot_model(bundle: DatasetBundle) -> Tuple[DotBinaryClassifier, Dict[str, List[float]], float]:
    print('\n' + '=' * 60)
    print(f'训练点积模型 ({bundle.config.name}, class {bundle.config.target_class} vs rest)')
    print('=' * 60)

    model = DotBinaryClassifier(bundle.config.input_dim).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(bundle.config.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for features, labels in bundle.train_loader:
            features = features.to(DEVICE)
            binary_labels = (labels == bundle.config.target_class).float().to(DEVICE)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, binary_labels)
            loss.backward()
            optimizer.step()

            predictions = (torch.sigmoid(logits) >= 0.5).float()
            running_loss += float(loss.item()) * binary_labels.size(0)
            running_correct += int((predictions == binary_labels).sum().item())
            running_total += int(binary_labels.size(0))

        train_loss = running_loss / running_total if running_total else 0.0
        train_acc = running_correct / running_total if running_total else 0.0
        test_acc = evaluate_binary(model, bundle.test_loader, bundle.config.target_class)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(test_acc)
        print(f'   epoch {epoch + 1}/{bundle.config.epochs}: loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}')

    final_acc = evaluate_binary(model, bundle.test_loader, bundle.config.target_class)
    return model, history, final_acc



def _select_feature_candidates(features: np.ndarray, limit: int = TREE_FEATURE_CANDIDATES) -> np.ndarray:
    variances = np.var(features, axis=0)
    if limit >= variances.shape[0]:
        return np.arange(variances.shape[0])
    top_indices = np.argpartition(variances, -limit)[-limit:]
    return np.sort(top_indices)



def train_decision_tree_stump(bundle: DatasetBundle) -> Tuple[Dict, float]:
    print('\n' + '=' * 60)
    print(f'训练决策树桩 ({bundle.config.name}, class {bundle.config.target_class} vs rest)')
    print('=' * 60)

    train_binary = (bundle.train_y == bundle.config.target_class).astype(np.int32)
    test_binary = (bundle.test_y == bundle.config.target_class).astype(np.int32)

    candidate_features = _select_feature_candidates(bundle.train_x, limit=min(TREE_FEATURE_CANDIDATES, bundle.config.input_dim))
    best_score = -1.0
    best_feature = 0
    best_threshold = 0.0
    best_left_value = 0.0
    best_right_value = 1.0

    for feature_idx in candidate_features:
        feature_values = bundle.train_x[:, feature_idx]
        thresholds = np.quantile(feature_values, TREE_THRESHOLD_QUANTILES)
        for threshold in np.unique(thresholds):
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue

            left_value = float(train_binary[left_mask].mean() >= 0.5)
            right_value = float(train_binary[right_mask].mean() >= 0.5)
            predictions = np.where(left_mask, left_value, right_value)
            score = float((predictions == train_binary).mean())
            if score > best_score:
                best_score = score
                best_feature = int(feature_idx)
                best_threshold = float(threshold)
                best_left_value = left_value
                best_right_value = right_value

    test_predictions = np.where(bundle.test_x[:, best_feature] <= best_threshold, best_left_value, best_right_value)
    test_accuracy = float((test_predictions == test_binary).mean())
    model = {
        'type': 'decision_tree',
        'format': 'dict',
        'root': 0,
        'target_class': bundle.config.target_class,
        'dataset_name': bundle.config.name,
        'nodes': [
            {'id': 0, 'feature': best_feature, 'threshold': best_threshold, 'left': 1, 'right': 2, 'is_leaf': False},
            {'id': 1, 'value': best_left_value, 'is_leaf': True},
            {'id': 2, 'value': best_right_value, 'is_leaf': True},
        ],
    }

    print(f'   best feature={best_feature}, threshold={best_threshold:.4f}, test_acc={test_accuracy:.4f}')
    return model, test_accuracy



def save_pickle(config: Dict, prefix: str, dataset_name: str) -> str:
    timestamp = config['timestamp']
    filepath = os.path.join(MODEL_SAVE_DIR, f'{prefix}_{dataset_name}_{timestamp}.pkl')
    with open(filepath, 'wb') as file_obj:
        pickle.dump(config, file_obj)
    print(f'✅ 模型保存: {filepath}')
    return filepath



def save_neural_network_model(model: FlattenLinearClassifier,
                              history: Dict[str, List[float]],
                              test_accuracy: float,
                              bundle: DatasetBundle,
                              timestamp: str) -> str:
    weights = model.linear.weight.detach().cpu().numpy()
    bias = model.linear.bias.detach().cpu().numpy()
    config = {
        'dataset_name': bundle.config.name,
        'model_name': 'cnn_single_layer_flattened',
        'timestamp': timestamp,
        'test_accuracy': test_accuracy,
        'architecture': {
            'dataset_name': bundle.config.name,
            'type': 'single_layer',
            'input_dim': bundle.config.input_dim,
            'output_dim': bundle.config.num_classes,
            'weights': weights.flatten().tolist(),
            'bias': bias.tolist(),
            'weights_shape': (bundle.config.num_classes, bundle.config.input_dim),
            'bias_shape': (bundle.config.num_classes,),
            'description': f'真实训练的单层线性 {bundle.config.name} 分类器',
        },
        'history_summary': history,
    }
    return save_pickle(config, 'cnn_flattened', bundle.config.name)



def save_dot_model(model: DotBinaryClassifier,
                   history: Dict[str, List[float]],
                   test_accuracy: float,
                   bundle: DatasetBundle,
                   timestamp: str) -> str:
    weights = model.linear.weight.detach().cpu().numpy().reshape(-1)
    config = {
        'dataset_name': bundle.config.name,
        'model_name': 'dot_binary_classifier',
        'timestamp': timestamp,
        'test_accuracy': test_accuracy,
        'architecture': {
            'dataset_name': bundle.config.name,
            'type': 'dot_product',
            'input_dim': bundle.config.input_dim,
            'weights': weights.tolist(),
            'target_class': bundle.config.target_class,
            'description': f'真实训练的点积模型，class {bundle.config.target_class} vs rest',
        },
        'history_summary': history,
    }
    return save_pickle(config, 'dot', bundle.config.name)



def save_decision_tree_model(model: Dict,
                             test_accuracy: float,
                             bundle: DatasetBundle,
                             timestamp: str) -> str:
    config = {
        'dataset_name': bundle.config.name,
        'model_name': 'decision_tree_stump',
        'timestamp': timestamp,
        'test_accuracy': test_accuracy,
        'architecture': model,
        'history_summary': {
            'note': '真实训练的二分类决策树桩',
            'target_class': bundle.config.target_class,
        },
    }
    return save_pickle(config, 'decision_tree', bundle.config.name)



def list_saved_models():
    print('\n' + '=' * 60)
    print('已保存的模型文件')
    print('=' * 60)

    model_files = [name for name in os.listdir(MODEL_SAVE_DIR) if name.endswith('.pkl')]
    if not model_files:
        print('   没有找到模型文件')
        return

    for index, filename in enumerate(sorted(model_files), start=1):
        filepath = os.path.join(MODEL_SAVE_DIR, filename)
        size_kb = os.path.getsize(filepath) / 1024
        print(f'   {index}. {filename} ({size_kb:.1f} KB)')



def train_dataset_models(dataset_name: str, data_dir: str) -> List[str]:
    bundle = build_dataset_bundle(dataset_name, data_dir)
    neural_model, neural_history, neural_acc = train_neural_network(bundle)
    dot_model, dot_history, dot_acc = train_dot_model(bundle)
    decision_tree_model, decision_tree_acc = train_decision_tree_stump(bundle)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return [
        save_neural_network_model(neural_model, neural_history, neural_acc, bundle, timestamp),
        save_dot_model(dot_model, dot_history, dot_acc, bundle, timestamp),
        save_decision_tree_model(decision_tree_model, decision_tree_acc, bundle, timestamp),
    ]



def main():
    parser = argparse.ArgumentParser(description='Train real experiment models for supported datasets.')
    parser.add_argument('--dataset', choices=['mnist', 'uci_har', 'all'], default='mnist', help='Dataset to train on')
    parser.add_argument('--data-dir', default='data', help='Dataset cache directory')
    args = parser.parse_args()

    print('=' * 80)
    print('真实模型训练脚本启动')
    print('=' * 80)
    print(f'模型保存目录: {MODEL_SAVE_DIR}')
    print(f'运行设备: {DEVICE}')

    set_seed()
    dataset_names = ['mnist', 'uci_har'] if args.dataset == 'all' else [args.dataset]
    saved_paths: List[str] = []
    for dataset_name in dataset_names:
        print('\n' + '#' * 72)
        print(f'训练数据集: {dataset_name}')
        print('#' * 72)
        saved_paths.extend(train_dataset_models(dataset_name, args.data_dir))

    print('\n' + '=' * 80)
    print('真实模型训练完成')
    print('=' * 80)
    for path in saved_paths:
        print(f'   {path}')
    list_saved_models()


if __name__ == '__main__':
    main()
