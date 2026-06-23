# decart/core/__init__.py
"""
DeCart core module package.
Contains all cryptographic primitives and basic mathematical operations.
"""

__version__ = "1.0.0"
__author__ = "DeCart Research Team"
__description__ = "Web 3.0 AI Query Decentralized Control Framework Core Modules"

# Export list
__all__ = [
    'BilinearPairing',
    'HomomorphicEncryption', 
    'FiniteField',
]

# Lazy import mapping
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
    """Lazy import function."""
    if name in _import_mapping:
        module_name, attr_name = _import_mapping[name]
        module = __import__(module_name, globals(), locals(), [attr_name], 1)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'decart.core' has no attribute '{name}'")

# Pre-import common modules into namespace
try:
    from .bilinear_pairing import BilinearPairing
    from .homomorphic import HomomorphicEncryption
    from .finite_field import FiniteField
    __all__.extend(['BilinearPairing', 'HomomorphicEncryption', 'FiniteField'])
except ImportError:
    # If import fails, use lazy import
    pass


def test_all_modules():
    """Convenience function to test all modules."""
    try:
        from .test_core import test_core_modules
        return test_core_modules()
    except ImportError:
        # If test_core does not exist, run basic test
        print("Please run python -m core.test_core for full tests")
        return False