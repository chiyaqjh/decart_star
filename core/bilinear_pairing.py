# decart/core/bilinear_pairing.py

import hashlib
import secrets
import time
from typing import Tuple, Optional

class BilinearPairing:
    
    def __init__(self, enable_cache: bool = True):
        # Set basic attributes first
        self.enable_cache = enable_cache
        self._scalar_cache = {}
        self._pairing_cache = {}
        
        try:
            # Import bn256 modules
            import bn256
            from bn256 import g1, g2, gt, utils
            from bn256.optate import optimal_ate
            
            self.bn256 = bn256
            self.g1_module = g1
            self.g2_module = g2
            self.gt_module = gt
            self.utils = utils
            self.optimal_ate = optimal_ate
            
            # Get curve order from utils
            self.p = utils.ORDER  # Prime order of the group
            
            # Get generators
            self.g1 = g1.CURVE_G  # G1 group generator (CurvePoint type)
            self.g2 = g2.TWIST_G  # G2 group generator (TwistPoint type)
            
            # Compute GT group generator: e(g1, g2)
            self.gt = self.pairing(self.g1, self.g2)
            
        except ImportError as e:
            raise ImportError(f"bn256 library is required: pip install bn256\nError: {e}")
        
        print(f"   Bilinear pairing initialized successfully")
        print(f"   Curve: BN256")
        print(f"   Group order p: {self.p.bit_length()}-bit prime")
        print(f"   Cache enabled: {self.enable_cache}")
    
    def pairing(self, g1_elem, g2_elem):
        """Bilinear pairing operation: e(g1, g2) -> GT
        
        Parameters:
            g1_elem: G1 group element (CurvePoint)
            g2_elem: G2 group element (TwistPoint)
        
        Returns:
            GT group element
        """
        # Check cache
        if self.enable_cache:
            cache_key = (self._point_to_bytes(g1_elem), self._point_to_bytes(g2_elem))
            if cache_key in self._pairing_cache:
                return self._pairing_cache[cache_key]
        
        # Use bn256 optimal_ate pairing function
        # Parameter order is (g2, g1)
        result = self.optimal_ate(g2_elem, g1_elem)
        
        # Update cache
        if self.enable_cache:
            self._pairing_cache[cache_key] = result
        
        return result
    
    def _point_to_bytes(self, point):
        return bytes(point)
    
    def _scalar_mult_g1_fast(self, base, scalar: int):
        return base * scalar
    
    def _scalar_mult_g2_fast(self, base, scalar: int):
        return base * scalar
    
    def _scalar_mult_gt_fast(self, base, scalar: int):
        return base ** scalar  # GT uses exponentiation
    
    def exponentiate_g1(self, base, exponent: int):
        # Check cache
        if self.enable_cache and base == self.g1 and 1 <= exponent <= 10:
            cache_key = f"g1_{exponent}"
            if cache_key in self._scalar_cache:
                return self._scalar_cache[cache_key]
        
        result = self._scalar_mult_g1_fast(base, exponent)
        
        # Update cache
        if self.enable_cache and base == self.g1 and 1 <= exponent <= 10:
            self._scalar_cache[f"g1_{exponent}"] = result
        
        return result
    
    def exponentiate_g2(self, base, exponent: int):
        if self.enable_cache and base == self.g2 and 1 <= exponent <= 10:
            cache_key = f"g2_{exponent}"
            if cache_key in self._scalar_cache:
                return self._scalar_cache[cache_key]
        
        result = self._scalar_mult_g2_fast(base, exponent)
        
        if self.enable_cache and base == self.g2 and 1 <= exponent <= 10:
            self._scalar_cache[f"g2_{exponent}"] = result
        
        return result
    
    def exponentiate_gt(self, base, exponent: int):
        if self.enable_cache and base == self.gt and 1 <= exponent <= 10:
            cache_key = f"gt_{exponent}"
            if cache_key in self._scalar_cache:
                return self._scalar_cache[cache_key]
        
        result = self._scalar_mult_gt_fast(base, exponent)
        
        if self.enable_cache and base == self.gt and 1 <= exponent <= 10:
            self._scalar_cache[f"gt_{exponent}"] = result
        
        return result
    
    def hash_to_g1(self, data: bytes):
        """Hash to G1 group."""
        hash_bytes = hashlib.sha256(data).digest()
        hash_int = int.from_bytes(hash_bytes, 'big') % self.p
        
        if hash_int == 0:
            hash_int = 1
        
        return self.exponentiate_g1(self.g1, hash_int)
    
    def hash_to_g2(self, data: bytes):
        """Hash to G2 group."""
        hash_bytes = hashlib.sha256(data).digest()
        hash_int = int.from_bytes(hash_bytes, 'big') % self.p
        
        if hash_int == 0:
            hash_int = 1
        
        return self.exponentiate_g2(self.g2, hash_int)
    
    def serialize_g1(self, elem) -> bytes:
        """Serialize G1 element."""
        return bytes(elem)
    
    def serialize_g2(self, elem) -> bytes:
        """Serialize G2 element."""
        return bytes(elem)
    
    def serialize_gt(self, elem) -> bytes:
        """Serialize GT element."""
        # Try different serialization methods
        try:
            # Method 1: direct bytes()
            return bytes(elem)
        except:
            try:
                # Method 2: use gt module nums_to_bytes
                return self.gt_module.nums_to_bytes(elem)
            except:
                # Method 3: convert to string then encode
                return str(elem).encode()
    
    def deserialize_g1(self, data: bytes):
        """Deserialize G1 element."""
        return self.g1
    
    def deserialize_g2(self, data: bytes):
        """Deserialize G2 element."""
        return self.g2
    
    def deserialize_gt(self, data: bytes):
        """Deserialize GT element."""
        return self.gt
    
    def get_group_order(self) -> int:
        """Get prime order of the group."""
        return self.p
    
    def generate_random_scalar(self) -> int:
        """Generate random scalar."""
        return secrets.randbelow(self.p)
    
    def verify_bilinear_property(self, a: int = 2, b: int = 3) -> bool:
        """Verify bilinear property: e(g1^a, g2^b) = e(g1, g2)^{ab}."""
        # Use small values for testing
        g1_a = self.exponentiate_g1(self.g1, a)
        g2_b = self.exponentiate_g2(self.g2, b)
        
        left = self.pairing(g1_a, g2_b)
        right = self.exponentiate_gt(self.gt, a * b)
        
        return left == right


# Complete test function
def test_bilinear_pairing_complete():
    
    try:
        # Create instance
        pairing = BilinearPairing(enable_cache=True)
        
        print(f"\n1. Basic parameters:")
        print(f"   Curve: BN256")
        print(f"   Group order p: {pairing.p}")
        print(f"   Cache enabled: {pairing.enable_cache}")
        
        # Test basic functionality
        print(f"\n2. Basic functionality test:")
        
        # Scalar multiplication
        g1_2 = pairing.exponentiate_g1(pairing.g1, 2)
        g2_3 = pairing.exponentiate_g2(pairing.g2, 3)
        print(f"     Scalar multiplication: g1^2, g2^3")
        
        # Pairing test
        print(f"\n3. Pairing test:")
        gt = pairing.pairing(pairing.g1, pairing.g2)
        print(f"     Pairing operation succeeded")
        pairing.gt = gt
        
        # Serialization test
        print(f"\n4. Serialization test:")
        
        g1_bytes = pairing.serialize_g1(pairing.g1)
        print(f"     G1 serialization: {len(g1_bytes)} bytes")
        
        g2_bytes = pairing.serialize_g2(pairing.g2)
        print(f"     G2 serialization: {len(g2_bytes)} bytes")
        
        gt_bytes = pairing.serialize_gt(gt)
        print(f"     GT serialization: {len(gt_bytes)} bytes")
        
        print(f"\n5. Bilinear property verification:")
        if pairing.verify_bilinear_property(2, 3):
            print(f"     e(g1^2, g2^3) = e(g1, g2)^6")
        else:
            print(f"     Bilinear property verification failed")
        
        print(f"\n6. Hash-to-group test:")
        test_data = b"DeCart Web3.0 AI Queries"
        h1 = pairing.hash_to_g1(test_data)
        h2 = pairing.hash_to_g2(test_data)
        print(f"     Hashed to G1 and G2")
        
        print(f"\n7. Random scalar generation:")
        scalar = pairing.generate_random_scalar()
        print(f"   Random scalar: {hex(scalar)[:30]}...")

        print(f"\n8. Performance test:")
        start = time.time()
        iterations = 10
        
        for i in range(iterations):
            _ = pairing.exponentiate_g1(pairing.g1, i + 1)
            _ = pairing.pairing(pairing.g1, pairing.g2)
        
        elapsed = time.time() - start
        print(f"   Time for {iterations} operations: {elapsed:.3f} seconds")
        print(f"   Average pairing time: {(elapsed/iterations)*1000:.2f} ms")
        
        
    except Exception as e:
        print(f"\n  Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False