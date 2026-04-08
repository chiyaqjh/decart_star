# decart/core/__init__.py
"""
DeCart 核心密码学模块
包含完全非模拟的双线性配对、同态加密和有限域运算
"""

from .bilinear_pairing import BilinearPairing
from .homomorphic import HomomorphicEncryption
from .finite_field import FiniteField

__all__ = [
    'BilinearPairing',
    'HomomorphicEncryption', 
    'FiniteField'
]

# 版本信息
__version__ = "1.0.0"
__description__ = "DeCart 核心密码学模块 - 论文研究专用"
