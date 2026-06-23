"""UCI HAR dataset loading and preprocessing helpers."""

import os
import time
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


UCI_HAR_DOWNLOAD_URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
UCI_HAR_ARCHIVE_NAME = 'uci_har_dataset.zip'
UCI_HAR_INNER_ARCHIVE_NAME = 'UCI HAR Dataset.zip'
UCI_HAR_EXTRACTED_DIR = 'UCI HAR Dataset'


class UCIHARDataLoader:
    def __init__(self,
                 data_dir: str = './data',
                 batch_size: int = 128,
                 download: bool = True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.download = download
        self.dataset_dir = os.path.join(data_dir, UCI_HAR_EXTRACTED_DIR)
        self.archive_path = os.path.join(data_dir, UCI_HAR_ARCHIVE_NAME)
        self.train_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None

        os.makedirs(data_dir, exist_ok=True)

        print(f'\n UCI HAR data loader initialized')
        print(f'   Data directory: {data_dir}')
        print(f'   Batch size: {batch_size}')

    def _ensure_downloaded(self):
        if os.path.isdir(self.dataset_dir):
            return
        if not self.download:
            raise FileNotFoundError(f'UCI HAR dataset not found under {self.dataset_dir}')

        print(f'   Downloading UCI HAR dataset: {UCI_HAR_DOWNLOAD_URL}')
        urllib.request.urlretrieve(UCI_HAR_DOWNLOAD_URL, self.archive_path)

        with zipfile.ZipFile(self.archive_path, 'r') as outer_zip:
            outer_zip.extractall(self.data_dir)

        inner_archive_path = os.path.join(self.data_dir, UCI_HAR_INNER_ARCHIVE_NAME)
        if not os.path.isfile(inner_archive_path):
            raise FileNotFoundError(f'Inner UCI HAR archive not found: {inner_archive_path}')

        with zipfile.ZipFile(inner_archive_path, 'r') as inner_zip:
            inner_zip.extractall(self.data_dir)

    def _load_split(self, split: str) -> TensorDataset:
        split_dir = os.path.join(self.dataset_dir, split)
        x_path = os.path.join(split_dir, f'X_{split}.txt')
        y_path = os.path.join(split_dir, f'y_{split}.txt')

        features = np.loadtxt(x_path, dtype=np.float32)
        labels = np.loadtxt(y_path, dtype=np.int64) - 1

        feature_tensor = torch.from_numpy(features)
        label_tensor = torch.from_numpy(labels)
        return TensorDataset(feature_tensor, label_tensor)

    def load_data(self) -> Dict[str, DataLoader]:

        start_time = time.time()
        self._ensure_downloaded()
        self.train_dataset = self._load_split('train')
        self.test_dataset = self._load_split('test')
        elapsed = time.time() - start_time

        print(f'   Dataset loading completed! Time: {elapsed:.2f}s')
        print(f'   Training set: {len(self.train_dataset)} records')
        print(f'   Test set: {len(self.test_dataset)} records')
        print(f'   Feature dimension: 561')
        print(f'   Number of classes: 6')

        return self.get_dataloaders()

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        loaders: Dict[str, DataLoader] = {}
        if self.train_dataset is not None:
            loaders['train'] = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        if self.test_dataset is not None:
            loaders['test'] = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return loaders

    def get_numpy_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if self.train_dataset is not None:
            train_x, train_y = self.train_dataset.tensors
            result['train'] = (train_x.numpy(), train_y.numpy())
        if self.test_dataset is not None:
            test_x, test_y = self.test_dataset.tensors
            result['test'] = (test_x.numpy(), test_y.numpy())
        return result

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            'name': 'UCI_HAR',
            'num_classes': 6,
            'input_dim': 561,
            'train_size': len(self.train_dataset) if self.train_dataset is not None else 0,
            'test_size': len(self.test_dataset) if self.test_dataset is not None else 0,
            'batch_size': self.batch_size,
        }



def load_uci_har(data_dir: str = './data', batch_size: int = 128) -> Dict[str, Any]:
    loader = UCIHARDataLoader(data_dir=data_dir, batch_size=batch_size)
    pytorch_loaders = loader.load_data()
    numpy_data = loader.get_numpy_data()
    return {
        'pytorch': pytorch_loaders,
        'numpy': numpy_data,
        'info': loader.get_dataset_info(),
        'loader': loader,
    }



def load_uci_har_records(num_records: int,
                         data_dir: str = './data',
                         split: str = 'test') -> Tuple[List[List[float]], List[int]]:
    if split not in {'train', 'test'}:
        raise ValueError("split must be 'train' or 'test'")

    loader = UCIHARDataLoader(data_dir=data_dir, batch_size=num_records)
    loader.load_data()
    dataset = loader.train_dataset if split == 'train' else loader.test_dataset
    if dataset is None:
        raise ValueError(f'UCI HAR {split} dataset is not available')

    count = min(num_records, len(dataset))
    records: List[List[float]] = []
    labels: List[int] = []
    for index in range(count):
        features, label = dataset[index]
        records.append(features.tolist())
        labels.append(int(label))

    if count < num_records:
        raise ValueError(f"Requested {num_records} UCI HAR records, but only found {count} in split '{split}'")

    return records, labels
