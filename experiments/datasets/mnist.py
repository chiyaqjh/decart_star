# decart/experiments/datasets/mnist.py

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import os
import time


class MNISTDataLoader:
    
    def __init__(self, 
                 data_dir: str = './data',
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 normalize: bool = True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        
        # Create the data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Define data transforms
        self.transform_train, self.transform_test = self._get_transforms()
        
        # Datasets
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        
        print(f"\n MNIST data loader initialized")
        print(f"   Data directory: {data_dir}")
        print(f"   Batch size: {batch_size}")
        print(f"   Normalization: {normalize}")
    
    def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Get data transforms.

        Returns:
            (train_transform, test_transform)
        """
        # Base transform: convert to tensor
        transform_list = [transforms.ToTensor()]
        
        # Normalize to [0,1]
        if self.normalize:
            # MNIST is [0,1] by default; normalize explicitly here
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        # Test-set transform (no augmentation)
        test_transform = transforms.Compose(transform_list.copy())
        
        # Training-set transform (with optional augmentation)
        train_transform_list = transform_list.copy()
        # Optional: add mild data augmentation
        # train_transform_list.insert(0, transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)))
        
        train_transform = transforms.Compose(train_transform_list)
        
        return train_transform, test_transform
    
    def load_data(self, 
                  use_validation: bool = True,
                  val_ratio: float = 0.1,
                  download: bool = True) -> Dict[str, DataLoader]:
        
        start_time = time.time()
        
        # Load training set
        self.train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=download,
            transform=self.transform_train
        )
        
        # Load test set
        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=download,
            transform=self.transform_test
        )
        
        load_time = time.time() - start_time
        
        print(f"   Dataset loading completed! Time: {load_time:.2f}s")
        print(f"   Training set: {len(self.train_dataset)} images")
        print(f"   Test set: {len(self.test_dataset)} images")
        print(f"   Image size: 28x28")
        print(f"   Number of classes: 10")
        
        # Split validation set
        if use_validation:
            val_size = int(len(self.train_dataset) * val_ratio)
            train_size = len(self.train_dataset) - val_size
            
            # Random split
            indices = list(range(len(self.train_dataset)))
            np.random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            # Create training subset
            self.train_dataset = Subset(self.train_dataset, train_indices)
            
            # Create validation set (using test transform)
            full_train_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=False,
                transform=self.transform_test
            )
            self.val_dataset = Subset(full_train_dataset, val_indices)
            
            print(f"   Validation set: {len(self.val_dataset)} images ({val_ratio*100:.0f}%)")
        
        # Create data loaders
        loaders = self.get_dataloaders()
        
        return loaders
    
    def get_dataloaders(self) -> Dict[str, DataLoader]:

        loaders = {}
        
        # Training loader
        if self.train_dataset is not None:
            loaders['train'] = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        
        # Test loader
        if self.test_dataset is not None:
            loaders['test'] = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        
        # Validation loader
        if self.val_dataset is not None:
            loaders['val'] = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        
        return loaders
    
    def get_numpy_data(self, flatten: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        result = {}
        
        # Training set
        if self.train_dataset is not None:
            X_train = []
            y_train = []
            
            # Handle raw dataset or Subset
            if isinstance(self.train_dataset, Subset):
                dataset = self.train_dataset.dataset
                indices = self.train_dataset.indices
                
                for idx in indices:
                    img, label = dataset[idx]
                    if flatten:
                        img = img.view(-1).numpy()
                    else:
                        img = img.numpy()
                    X_train.append(img)
                    y_train.append(label)
            else:
                for i in range(len(self.train_dataset)):
                    img, label = self.train_dataset[i]
                    if flatten:
                        img = img.view(-1).numpy()
                    else:
                        img = img.numpy()
                    X_train.append(img)
                    y_train.append(label)
            
            result['train'] = (np.array(X_train), np.array(y_train))
        
        # Test set
        if self.test_dataset is not None:
            X_test = []
            y_test = []
            
            for i in range(len(self.test_dataset)):
                img, label = self.test_dataset[i]
                if flatten:
                    img = img.view(-1).numpy()
                else:
                    img = img.numpy()
                X_test.append(img)
                y_test.append(label)
            
            result['test'] = (np.array(X_test), np.array(y_test))
        
        # Validation set
        if self.val_dataset is not None:
            X_val = []
            y_val = []
            
            if isinstance(self.val_dataset, Subset):
                dataset = self.val_dataset.dataset
                indices = self.val_dataset.indices
                
                for idx in indices:
                    img, label = dataset[idx]
                    if flatten:
                        img = img.view(-1).numpy()
                    else:
                        img = img.numpy()
                    X_val.append(img)
                    y_val.append(label)
            else:
                for i in range(len(self.val_dataset)):
                    img, label = self.val_dataset[i]
                    if flatten:
                        img = img.view(-1).numpy()
                    else:
                        img = img.numpy()
                    X_val.append(img)
                    y_val.append(label)
            
            result['val'] = (np.array(X_val), np.array(y_val))
        
        return result
    
    def get_sample_batch(self, split: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        loaders = self.get_dataloaders()
        
        if split not in loaders:
            raise ValueError(f"Invalid dataset split: {split}")
        
        loader = loaders[split]
        data, targets = next(iter(loader))
        
        return data, targets
    
    def get_dataset_info(self) -> Dict[str, Any]:

        info = {
            'name': 'MNIST',
            'num_classes': 10,
            'image_size': (1, 28, 28),
            'input_dim': 784,
            'train_size': len(self.train_dataset) if self.train_dataset else 0,
            'test_size': len(self.test_dataset) if self.test_dataset else 0,
            'val_size': len(self.val_dataset) if self.val_dataset else 0,
            'batch_size': self.batch_size,
            'normalized': self.normalize
        }
        
        return info


#  Convenience functions

def load_mnist(data_dir: str = './data',
               batch_size: int = 64,
               use_validation: bool = True,
               val_ratio: float = 0.1,
               flatten_for_svm: bool = False) -> Dict[str, Any]:

    # Create loader
    loader = MNISTDataLoader(
        data_dir=data_dir,
        batch_size=batch_size
    )
    
    # Load data
    pytorch_loaders = loader.load_data(
        use_validation=use_validation,
        val_ratio=val_ratio
    )
    
    # Get NumPy format (for SVM)
    numpy_data = loader.get_numpy_data(flatten=True)
    
    result = {
        'pytorch': pytorch_loaders,
        'numpy': numpy_data,
        'info': loader.get_dataset_info(),
        'loader': loader
    }
    
    return result


def load_mnist_records(num_records: int,
                       data_dir: str = './data',
                       split: str = 'test') -> Tuple[List[List[float]], List[int]]:
    
    if split not in {'train', 'test'}:
        raise ValueError("split must be 'train' or 'test'")

    loader = MNISTDataLoader(data_dir=data_dir, batch_size=num_records)
    loader.load_data(use_validation=False, download=True)

    dataset = loader.train_dataset if split == 'train' else loader.test_dataset
    if dataset is None:
        raise ValueError(f"MNIST {split} dataset is not available")

    count = min(num_records, len(dataset))
    records = []
    labels = []
    for index in range(count):
        image, label = dataset[index]
        records.append(image.view(-1).tolist())
        labels.append(int(label))

    if count < num_records:
        raise ValueError(f"Requested {num_records} MNIST records, but only found {count} in split '{split}'")

    return records, labels


def visualize_mnist_sample(loader: MNISTDataLoader, 
                          num_samples: int = 5,
                          save_path: Optional[str] = None):

    try:
        import matplotlib.pyplot as plt
        
        # Get one batch of data
        data, targets = loader.get_sample_batch('train')
        
        fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
        
        for i in range(num_samples):
            img = data[i].squeeze().numpy()
            label = targets[i].item()
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f" Image saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print(" matplotlib must be installed for visualization")


#  Test code

def test_mnist_loader():
    
    try:
        # Create a temporary data directory
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        print(f"1. Initialize loader...")
        loader = MNISTDataLoader(
            data_dir=temp_dir,
            batch_size=32,
            num_workers=0
        )
        
        print(f"\n2. Load data (using download=True)...")
        loaders = loader.load_data(
            use_validation=True,
            val_ratio=0.1,
            download=True
        )
        
        # Validate dataset sizes
        info = loader.get_dataset_info()
        print(f"\n3. Dataset information:")
        print(f"   Training set: {info['train_size']} images")
        print(f"   Validation set: {info['val_size']} images")
        print(f"   Test set: {info['test_size']} images")
        
        assert info['train_size'] > 0, "Training set is empty"
        assert info['test_size'] > 0, "Test set is empty"
        if info['val_size'] > 0:
            print(f"   Validation split succeeded")
        
        print(f"\n4. Test data loaders...")
        for split_name, loader_obj in loaders.items():
            data, targets = next(iter(loader_obj))
            print(f"   {split_name}: batch shape {data.shape}, label shape {targets.shape}")
            assert data.shape[0] == 32, f"Incorrect batch size: {data.shape[0]}"
        
        print(f"\n5. Test NumPy data format...")
        numpy_data = loader.get_numpy_data(flatten=True)
        
        for split_name, (X, y) in numpy_data.items():
            print(f"   {split_name}: X shape {X.shape}, y shape {y.shape}")
            if split_name != 'test':  # The test split may not align as precisely as the validation split
                assert X.shape[1] == 784, f"Incorrect flattened dimension: {X.shape[1]}"
        
        print(f"\n All MNIST data loader tests passed!")
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    test_mnist_loader()
    