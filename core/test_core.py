# decart/core/__init__.py
"""
DeCart 核心模块包
包含所有密码学原语和基础数学运算
"""

__version__ = "1.0.0"
__author__ = "DeCart Research Team"
__description__ = "Web 3.0 AI Query Decentralized Control Framework Core Modules"

# 导出列表
__all__ = [
    'BilinearPairing',
    'HomomorphicEncryption', 
    'FiniteField',
]

# 延迟导入的映射
_import_mapping = {
    'BilinearPairing': ('.bilinear_pairing', 'BilinearPairing'),
    'test_bilinear_pairing_complete': ('.bilinear_pairing', 'test_bilinear_pairing_complete'),
    'HomomorphicEncryption': ('.homomorphic', 'HomomorphicEncryption'),
    'test_homomorphic_simple': ('.homomorphic', 'test_homomorphic_simple'),
    'FiniteField': ('.finite_field', 'FiniteField'),
    'test_finite_field_basic': ('.finite_field', 'test_finite_field_basic'),
    'test_finite_field_advanced': ('.finite_field', 'test_finite_field_advanced'),
}

def __getattr__(name):
    """延迟导入函数"""
    if name in _import_mapping:
        module_name, attr_name = _import_mapping[name]
        module = __import__(module_name, globals(), locals(), [attr_name], 1)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'decart.core' has no attribute '{name}'")

# 预导入一些常用模块到命名空间
try:
    from .bilinear_pairing import BilinearPairing
    from .homomorphic import HomomorphicEncryption
    from .finite_field import FiniteField
    __all__.extend(['BilinearPairing', 'HomomorphicEncryption', 'FiniteField'])
except ImportError:
    # 如果导入失败，使用延迟导入
    pass


def test_all_modules():
    """测试所有模块的便捷函数"""
    try:
        from .test_core import test_core_modules
        return test_core_modules()
    except ImportError:
        # 如果test_core不存在，运行基本测试
        print("请运行 python -m core.test_core 进行完整测试")
        return False