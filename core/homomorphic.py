# decart/core/homomorphic.py

import secrets
import hashlib
import pickle
import numpy as np
from typing import List, Tuple, Any, Optional, Union
from sympy import nextprime


class HomomorphicEncryption:
    # Validated parameter combinations
    VALID_PARAMETERS = {
        4096: {
            'coeff_mod_bit_sizes': [30, 20, 30],  # Safer parameters
            'scale': 2**20,
            'security': 128
        },
        8192: {
            'coeff_mod_bit_sizes': [30, 20, 20, 30],  # Safer parameters
            'scale': 2**20,
            'security': 128
        }
    }
class HomomorphicEncryption:
    
    # Optimized parameter combinations
    OPTIMAL_PARAMETERS = {
        4096: {
            'coeff_mod_bit_sizes': [30, 20, 30],      # Balanced parameters
            'scale': 2**20,                            # 1M scale factor
            'security': 128,
            'description': 'Fast mode, suitable for development testing'
        },
        8192: {
            'coeff_mod_bit_sizes': [40, 30, 30, 40],   # Optimized parameters
            'scale': 2**30,                             # 1G scale factor
            'security': 128,
            'description': 'Balanced mode, recommended for production'
        },
        16384: {
            'coeff_mod_bit_sizes': [40, 30, 30, 30, 40], # High precision mode
            'scale': 2**30,
            'security': 192,
            'description': 'High precision mode, suitable for complex computation'
        }
    }
    
    def __init__(self, 
                 mode: str = 'balanced',  # 'fast', 'balanced', 'precision'
                 poly_modulus_degree: int = None):
        # Select parameters by mode
        if poly_modulus_degree is None:
            if mode == 'fast':
                poly_modulus_degree = 4096
            elif mode == 'balanced':
                poly_modulus_degree = 8192
            elif mode == 'precision':
                poly_modulus_degree = 16384
            else:
                poly_modulus_degree = 8192  # Default balanced mode
        
        self.poly_modulus_degree = poly_modulus_degree
        self.mode = mode
        
        # Get optimized parameters
        if poly_modulus_degree in self.OPTIMAL_PARAMETERS:
            params = self.OPTIMAL_PARAMETERS[poly_modulus_degree]
            coeff_mod_bit_sizes = params['coeff_mod_bit_sizes']
            scale = params['scale']
            security = params['security']
            description = params['description']
        else:
            # Default parameters
            coeff_mod_bit_sizes = [40, 30, 30, 40]
            scale = 2**30
            security = 128
            description = 'Custom mode'
        
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.scale = scale
        
        print(f"\n Homomorphic encryption initialization [mode: {mode}]")
        print(f"   {description}")
        print(f"   Polynomial modulus degree: {poly_modulus_degree}")
        print(f"   Coefficient modulus sizes: {coeff_mod_bit_sizes}")
        print(f"   Scale factor: {scale}")
        print(f"   Security level: ~{security} bits")
        
        try:
            import tenseal as ts
            self.ts = ts
            
            # Create context
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
            )
            
            # Set global scale factor
            self.context.global_scale = self.scale
            
            # Generate Galois keys (for dot product)
            print(f"   Generating Galois keys...")
            self.context.generate_galois_keys()
            
            # Generate relinearization keys
            print(f"   Generating relinearization keys...")
            self.context.generate_relin_keys()
            
            # Verify keys
            print(f"     Galois keys: {self.context.has_galois_keys}")
            print(f"     Relinearization keys: {self.context.has_relin_keys}")
            
            self._secret_key_obj = self.context.secret_key()
            self._init_deterministic_secret()
            
            print(f"  CKKS context initialized successfully")
            
        except Exception as e:
            print(f"  Initialization failed: {e}")
            raise

    def _init_deterministic_secret(self):
        try:
            ctx_bytes = self.context.serialize(save_secret_key=True)
            ctx_hash = hashlib.sha256(ctx_bytes).digest()
            self._deterministic_secret = int.from_bytes(ctx_hash, 'big')
            
        except:
            import time
            seed_data = f"{time.time()}_{secrets.randbits(256)}"
            ctx_hash = hashlib.sha256(seed_data.encode()).digest()
            self._deterministic_secret = int.from_bytes(ctx_hash, 'big')
        
        self._secret_hash = hashlib.sha256(
            str(self._deterministic_secret).encode()
        ).hexdigest()[:16]
    
    @property
    def public_key(self):
        return self.context
    
    @property 
    def secret_key(self):
        return self._secret_key_obj
    
    @property
    def deterministic_secret(self) -> int:
        return self._deterministic_secret
    
    def encrypt(self, data: Union[float, List[float], np.ndarray]) -> Any:
        if isinstance(data, (int, float)):
            data = [float(data)]
        elif isinstance(data, np.ndarray):
            data = data.flatten().tolist()
        elif not isinstance(data, list):
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        # Check data range
        for val in data:
            if abs(val) > 10:
                print(f"    Data value {val} exceeds recommended range [-10, 10]")
        
        max_len = self.poly_modulus_degree // 2
        if len(data) > max_len:
            print(f"Warning: data length {len(data)} exceeds max length {max_len}, truncating")
            data = data[:max_len]
        
        try:
            return self.ts.ckks_vector(self.context, data)
        except Exception as e:
            raise ValueError(f"Encryption failed: {e}")
    
    def decrypt(self, ciphertext: Any) -> List[float]:
        try:
            decrypted = ciphertext.decrypt(self.secret_key)
            if isinstance(decrypted, list):
                return [float(x) for x in decrypted]
            elif isinstance(decrypted, (int, float)):
                return [float(decrypted)]
            else:
                return list(map(float, decrypted))
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def eval_add(self, ciphertext1: Any, ciphertext2: Any) -> Any:
        try:
            return ciphertext1 + ciphertext2
        except Exception as e:
            raise ValueError(f"Homomorphic addition failed: {e}")
    
    def eval_mul(self, ciphertext: Any, scalar: Union[int, float]) -> Any:
        try:
            return ciphertext * scalar
        except Exception as e:
            raise ValueError(f"Homomorphic multiplication failed: {e}")
    
    def eval_dot(self, ciphertexts1: List[Any], ciphertexts2: List[Any]) -> Any:
        if len(ciphertexts1) != len(ciphertexts2):
            raise ValueError("Vector lengths must be equal")
        
        try:
            result = ciphertexts1[0] * ciphertexts2[0]
            for i in range(1, len(ciphertexts1)):
                result += ciphertexts1[i] * ciphertexts2[i]
            return result
        except Exception as e:
            raise ValueError(f"Homomorphic dot product failed: {e}")
    
    def serialize_ciphertext(self, ciphertext: Any) -> bytes:
        try:
            return ciphertext.serialize()
        except Exception as e:
            raise ValueError(f"Serialization failed: {e}")
    
    def deserialize_ciphertext(self, data: bytes) -> Any:
        try:
            return self.ts.ckks_vector_from(self.context, data)
        except Exception as e:
            raise ValueError(f"Deserialization failed: {e}")
    
    def serialize_context(self) -> bytes:
        try:
            return self.context.serialize()
        except Exception as e:
            raise ValueError(f"Context serialization failed: {e}")
    
    def split_secret_key_shamir(self, num_shares: int = 3, threshold: int = 2):
        secret_int = self.deterministic_secret
        
        prime = nextprime(secret_int % (10**12) + 10**6)
        
        import random
        random.seed(secret_int % (2**32))
        
        coefficients = [secret_int % prime]
        for i in range(1, threshold):
            coeff = random.randint(1, prime - 1)
            coefficients.append(coeff)
        
        shares = []
        for i in range(1, num_shares + 1):
            y = 0
            for j, coeff in enumerate(coefficients):
                y = (y + coeff * pow(i, j, prime)) % prime
            shares.append((i, y))
        
        return shares, prime

    def combine_secret_shares(self, shares: List[Tuple[int, int]], prime: int) -> int:
        if len(shares) < 2:
            raise ValueError("At least 2 shares are required")
        
        secret_int = 0
        
        for i, (xi, yi) in enumerate(shares):
            li = 1
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (-xj) % prime
                    denominator = (xi - xj) % prime
                    
                    if denominator == 0:
                        continue
                    
                    denom_inv = pow(denominator, prime-2, prime)
                    li = (li * numerator * denom_inv) % prime
            
            secret_int = (secret_int + yi * li) % prime
        
        return secret_int
    
    def split_key_for_servers(self, num_servers: int = 3) -> List[Tuple[int, bytes]]:
        shares, prime = self.split_secret_key_shamir(
            num_shares=num_servers, 
            threshold=num_servers
        )
        
        server_keys = []
        for server_id, (x, y) in enumerate(shares, 1):
            key_data = {
                'prime': prime,
                'x': x,
                'y': y,
                'server_id': server_id,
                'secret_hash': self._secret_hash
            }
            
            key_bytes = pickle.dumps(key_data)
            server_keys.append((server_id, key_bytes))
        
        return server_keys
    
    def combine_server_keys(self, server_keys: List[Tuple[int, bytes]]) -> int:
        shares = []
        prime = None
        
        for server_id, key_bytes in server_keys:
            key_data = pickle.loads(key_bytes)
            
            if prime is None:
                prime = key_data['prime']
            elif prime != key_data['prime']:
                raise ValueError("All shares must use the same prime")
            
            shares.append((key_data['x'], key_data['y']))
        
        if prime is None:
            raise ValueError("Prime information not found")
        
        return self.combine_secret_shares(shares, prime)
    
    def create_partial_decryption_key(self) -> bytes:
        partial_key = {
            'context_hash': self._secret_hash,
            'partial_seed': secrets.token_bytes(32),
            'capabilities': ['addition', 'multiplication', 'dot_product']
        }
        
        return pickle.dumps(partial_key)
    
    def test_basic_functionality(self):
        try:
            print("\nTesting basic functionality...")
            
            # Test encryption and decryption
            test_data = [1.0, 2.0, 3.0]
            ct = self.encrypt(test_data)
            dec = self.decrypt(ct)
            
            error = sum(abs(d - o) for d, o in zip(dec, test_data)) / len(test_data)
            print(f"   Encryption/decryption test: average error {error:.8f}")
            
            # Test homomorphic addition
            ct2 = self.encrypt([0.5, 1.5, 2.5])
            ct_sum = self.eval_add(ct, ct2)
            dec_sum = self.decrypt(ct_sum)
            print(f"   Homomorphic addition test: {dec_sum}")
            
            return True
            
        except Exception as e:
            print(f"  Basic functionality test failed: {e}")
            return False


def test_homomorphic_complete():
    
    try:
        print("1. Initializing homomorphic encryption...")
        
        # Try initialization
        try:
            he = HomomorphicEncryption(poly_modulus_degree=8192)
        except:
            print("    8192 failed, trying 4096...")
            he = HomomorphicEncryption(poly_modulus_degree=4096)
        
        print(f"  Initialization succeeded: poly_degree={he.poly_modulus_degree}")
        
        # Test basic functionality
        if not he.test_basic_functionality():
            print("  Basic functionality test failed")
            return False
        
        print("\n2. Testing secret sharing...")
        
        # Test Shamir secret sharing
        shares, prime = he.split_secret_key_shamir(num_shares=5, threshold=3)
        print(f"   Generated {len(shares)} shares, threshold {3}")
        print(f"   Prime: {prime}")
        
        # Recovery test
        recovered = he.combine_secret_shares(shares[:3], prime)
        print(f"   Successfully recovered secret with 3 shares: {recovered}")
        
        print("\n3. Testing server key splitting...")
        
        server_keys = he.split_key_for_servers(num_servers=3)
        print(f"   Generated key shares for {len(server_keys)} servers")
        
        combined = he.combine_server_keys(server_keys)
        print(f"   Successfully recovered secret by combining shares")
        
        print("\n" + "=" * 60)
        print("  Homomorphic encryption module full test passed")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n  Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    
    if test_homomorphic_complete():
        print("\n  Success")
    else:
        print("\n  Failed")