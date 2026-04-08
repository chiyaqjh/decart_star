# decart/experiments/datasets/__init__.py
"""
DeCart 实验 - 数据集模块
提供 MNIST 数据集的加载和预处理
"""

from .mnist import MNISTDataLoader, load_mnist, visualize_mnist_sample

__all__ = [
    'MNISTDataLoader',
    'load_mnist',
    'visualize_mnist_sample'
]

__version__ = "1.0.0"
__description__ = "DeCart 实验数据集库 - MNIST"

print(f"✅ 加载 DeCart 实验数据集模块 v{__version__}")
print(f"   可用数据集: MNIST")