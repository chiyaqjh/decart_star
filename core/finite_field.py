# decart/core/finite_field.py

import secrets
import hashlib
import math
from typing import Tuple, List, Optional, Union
from sympy import isprime, nextprime


class FiniteField:
    
    def __init__(self, p: Optional[int] = None, bit_length: int = 256):
        if p is not None:
            if not isprime(p):
                raise ValueError(f"{p} is not a prime")
            self.p = p
            self.bit_length = p.bit_length()
        else:
            # Generate a secure prime
            self.p = self._generate_prime(bit_length)
            self.bit_length = bit_length
        
        print(f"  Finite field initialized successfully")
        print(f"   Prime modulus p: {self.p}")
        print(f"   Bit length: {self.bit_length}")
        print(f"   Hex: 0x{self.p:X}")
    
    def _generate_prime(self, bit_length: int) -> int:
        if bit_length < 128:
            raise ValueError("Prime bit length must be at least 128 bits for security")
        
        print(f"   Generating a {bit_length}-bit prime...")
        
        # Use a deterministic method to generate prime (for research reproducibility)
        seed = b"DeCart_Finite_Field_Research"
        base = int.from_bytes(hashlib.sha256(seed).digest(), 'big')
        
        # Ensure it is odd
        if base % 2 == 0:
            base += 1
        
        # Find the next prime
        prime_candidate = nextprime(base >> (base.bit_length() - bit_length))
        
        # Verify primality
        if not isprime(prime_candidate):
            # If it fails, use a random method
            prime_candidate = secrets.randbits(bit_length)
            if prime_candidate % 2 == 0:
                prime_candidate += 1
            
            while not isprime(prime_candidate):
                prime_candidate += 2
                # Prevent infinite loop
                if prime_candidate.bit_length() > bit_length + 10:
                    prime_candidate = secrets.randbits(bit_length)
                    if prime_candidate % 2 == 0:
                        prime_candidate += 1
        
        return prime_candidate
    
    def random_element(self) -> int:
        return secrets.randbelow(self.p)
    
    def random_nonzero(self) -> int:
        element = self.random_element()
        while element == 0:
            element = self.random_element()
        return element
    
    def add(self, a: int, b: int) -> int:
        return (a + b) % self.p
    
    def sub(self, a: int, b: int) -> int:
        return (a - b) % self.p
    
    def mul(self, a: int, b: int) -> int:
        return (a * b) % self.p
    
    def pow(self, base: int, exponent: int) -> int:
        return pow(base, exponent, self.p)
    
    def inv(self, a: int) -> int:
        if a == 0:
            raise ValueError("0 has no modular inverse")
        
        # Use the extended Euclidean algorithm
        return pow(a, self.p - 2, self.p)  # Fermat's little theorem: a^{p-1} ≡ 1 mod p
    
    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ValueError("Division by zero")
        
        b_inv = self.inv(b)
        return self.mul(a, b_inv)
    
    def sqrt(self, a: int) -> Tuple[Optional[int], Optional[int]]:
        if a == 0:
            return 0, 0
        
        # Check whether a square root exists (Euler's criterion)
        legendre = pow(a, (self.p - 1) // 2, self.p)
        if legendre != 1:
            return None, None  # No square root
        
        # Compute square root using the Tonelli-Shanks algorithm
        # Special case: p ≡ 3 mod 4
        if self.p % 4 == 3:
            root = pow(a, (self.p + 1) // 4, self.p)
            return root, (self.p - root) % self.p
        
        # General case: Tonelli-Shanks algorithm
        return self._tonelli_shanks(a)
    
    def _tonelli_shanks(self, n: int) -> Tuple[int, int]:
        Q = self.p - 1
        S = 0
        while Q % 2 == 0:
            Q //= 2
            S += 1
        
        # Find a quadratic non-residue
        z = 2
        while pow(z, (self.p - 1) // 2, self.p) != self.p - 1:
            z += 1
        
        M = S
        c = pow(z, Q, self.p)
        t = pow(n, Q, self.p)
        R = pow(n, (Q + 1) // 2, self.p)
        
        while t != 1:
            # Find the smallest i such that t^{2^i} ≡ 1
            i = 1
            t2i = pow(t, 2, self.p)
            while t2i != 1:
                t2i = pow(t2i, 2, self.p)
                i += 1
            
            b = pow(c, 1 << (M - i - 1), self.p)
            M = i
            c = pow(b, 2, self.p)
            t = self.mul(t, c)
            R = self.mul(R, b)
        
        return R, (self.p - R) % self.p
    
    def legendre_symbol(self, a: int) -> int:
        ls = pow(a, (self.p - 1) // 2, self.p)
        if ls == self.p - 1:
            return -1
        return ls
    
    def is_quadratic_residue(self, a: int) -> bool:
        return self.legendre_symbol(a) == 1
    
    def batch_add(self, elements: List[int]) -> int:
        result = 0
        for elem in elements:
            result = self.add(result, elem)
        return result
    
    def batch_mul(self, elements: List[int]) -> int:
        result = 1
        for elem in elements:
            result = self.mul(result, elem)
        return result
    
    def dot_product(self, vec1: List[int], vec2: List[int]) -> int:
        if len(vec1) != len(vec2):
            raise ValueError("Vector lengths must be equal")
        
        result = 0
        for a, b in zip(vec1, vec2):
            product = self.mul(a, b)
            result = self.add(result, product)
        return result
    
    def matrix_mul(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        rows1, cols1 = len(mat1), len(mat1[0])
        rows2, cols2 = len(mat2), len(mat2[0])
        
        if cols1 != rows2:
            raise ValueError("Matrix dimensions do not match")
        
        result = [[0] * cols2 for _ in range(rows1)]
        
        for i in range(rows1):
            for j in range(cols2):
                for k in range(cols1):
                    product = self.mul(mat1[i][k], mat2[k][j])
                    result[i][j] = self.add(result[i][j], product)
        
        return result
    
    def random_vector(self, n: int) -> List[int]:
        return [self.random_element() for _ in range(n)]
    
    def random_matrix(self, rows: int, cols: int) -> List[List[int]]:
        return [[self.random_element() for _ in range(cols)] for _ in range(rows)]
    
    def evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        result = 0
        power = 1
        
        for coeff in coefficients:
            term = self.mul(coeff, power)
            result = self.add(result, term)
            power = self.mul(power, x)
        
        return result
    
    def lagrange_interpolation(self, points: List[Tuple[int, int]]) -> List[int]:
        n = len(points)
        
        # Initialize coefficients to 0
        coefficients = [0] * n
        
        for i in range(n):
            xi, yi = points[i]
            
            # Compute Lagrange basis polynomial li(x)
            li = [1]  # Initial polynomial: 1
            
            for j in range(n):
                if i != j:
                    xj, _ = points[j]
                    
                    # Multiply by (x - xj)/(xi - xj)
                    denominator = self.sub(xi, xj)
                    denom_inv = self.inv(denominator)
                    
                    # Polynomial multiply by (x - xj)
                    new_li = [0] * (len(li) + 1)
                    for k in range(len(li)):
                        # Multiply by -xj
                        term1 = self.mul(li[k], self.p - xj)
                        new_li[k] = self.add(new_li[k], term1)
                        
                        # Multiply by x
                        term2 = li[k]
                        new_li[k + 1] = self.add(new_li[k + 1], term2)
                    
                    li = new_li
                    
                    # Multiply by 1/(xi - xj)
                    for k in range(len(li)):
                        li[k] = self.mul(li[k], denom_inv)
            
            # Multiply basis polynomial by yi and add to result
            for k in range(len(li)):
                term = self.mul(li[k], yi)
                coefficients[k] = self.add(coefficients[k], term)
        
        return coefficients
    
    def to_bytes(self, element: int) -> bytes:
        byte_length = (self.p.bit_length() + 7) // 8
        return element.to_bytes(byte_length, 'big')
    
    def from_bytes(self, data: bytes) -> int:
        element = int.from_bytes(data, 'big')
        return element % self.p
    
    def hash_to_field(self, data: bytes) -> int:
        hash_bytes = hashlib.sha256(data).digest()
        element = int.from_bytes(hash_bytes, 'big')
        return element % self.p
    
    def __str__(self) -> str:
        return f"FiniteField(Z_{self.p})"
    
    def __repr__(self) -> str:
        return f"FiniteField(p=0x{self.p:X}, bit_length={self.bit_length})"


def test_finite_field_basic():
    
    try:
        # Use a small prime for easier testing
        print("1. Initialize finite field (small prime for easy testing)...")
        ff = FiniteField(p=65537)  # Use fixed prime to ensure test consistency
        print(f"   Field: Z_{ff.p}")
        
        # Test basic operations
        print("\n2. Testing basic operations...")
        
        a, b = 12345, 54321
        
        # Addition
        add_result = ff.add(a, b)
        expected_add = (a + b) % ff.p
        print(f"   Addition: {a} + {b} = {add_result} (expected: {expected_add}) {' ' if add_result == expected_add else ' '}")
        
        # Subtraction
        sub_result = ff.sub(a, b)
        expected_sub = (a - b) % ff.p
        print(f"   Subtraction: {a} - {b} = {sub_result} (expected: {expected_sub}) {' ' if sub_result == expected_sub else ' '}")
        
        # Multiplication
        mul_result = ff.mul(a, b)
        expected_mul = (a * b) % ff.p
        print(f"   Multiplication: {a} × {b} = {mul_result} (expected: {expected_mul}) {' ' if mul_result == expected_mul else ' '}")
        
        # Modular inverse
        inv_result = ff.inv(a)
        check_inv = ff.mul(a, inv_result)
        print(f"   Modular inverse: {a}⁻¹ = {inv_result}, verify: {a} × {inv_result} = {check_inv} {' ' if check_inv == 1 else ' '}")
        
        # Division
        div_result = ff.div(a, b)
        check_div = ff.mul(div_result, b)
        print(f"   Division: {a} ÷ {b} = {div_result}, verify: {div_result} × {b} = {check_div} {' ' if check_div == a else ' '}")
        
        # Exponentiation
        exp_result = ff.pow(a, 3)
        expected_pow = pow(a, 3, ff.p)
        print(f"   Exponentiation: {a}³ = {exp_result} (expected: {expected_pow}) {' ' if exp_result == expected_pow else ' '}")
        
        return True
        
    except Exception as e:
        print(f"\n  Basic operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finite_field_advanced():

    try:
        # Use a slightly larger prime
        ff = FiniteField(p=104729)  # The 10000th prime
        
        print("1. Testing quadratic residues and square roots...")
        
        # Test square roots
        test_cases = [4, 9, 16, 25, 100]
        for n in test_cases:
            root1, root2 = ff.sqrt(n)
            if root1 is not None:
                check1 = ff.pow(root1, 2)
                check2 = ff.pow(root2, 2) if root2 is not None else None
                print(f"   √{n} = {root1}, {root2}")
                print(f"   Verify: {root1}² = {check1}, {root2}² = {check2}")
        
        print("\n2. Testing batch operations...")
        
        # Batch addition
        numbers = [1, 2, 3, 4, 5]
        batch_sum = ff.batch_add(numbers)
        manual_sum = sum(numbers) % ff.p
        print(f"   Batch addition: {numbers} -> {batch_sum} (expected: {manual_sum})")
        
        # Batch multiplication
        batch_prod = ff.batch_mul(numbers)
        manual_prod = 1
        for n in numbers:
            manual_prod = (manual_prod * n) % ff.p
        print(f"   Batch multiplication: {numbers} -> {batch_prod} (expected: {manual_prod})")
        
        print("\n3. Testing vector and matrix operations...")
        
        # Vector dot product
        vec1 = [1, 2, 3]
        vec2 = [4, 5, 6]
        dot = ff.dot_product(vec1, vec2)
        manual_dot = sum(a*b for a,b in zip(vec1, vec2)) % ff.p
        print(f"   Dot product: {vec1}·{vec2} = {dot} (expected: {manual_dot})")
        
        # Matrix multiplication
        mat1 = [[1, 2], [3, 4]]
        mat2 = [[5, 6], [7, 8]]
        mat_result = ff.matrix_mul(mat1, mat2)
        print(f"   Matrix multiplication: {mat1} × {mat2} = {mat_result}")
        
        print("\n4. Testing Lagrange interpolation...")
        
        # Test interpolation (basis of Shamir secret sharing)
        points = [(1, 5), (2, 10), (3, 19)]  # Corresponding polynomial: 2x² + 3
        coeffs = ff.lagrange_interpolation(points)
        print(f"   Interpolation points: {points}")
        print(f"   Recovered coefficients: {coeffs}")
        
        # Verify interpolation
        for x, y in points:
            poly_val = ff.evaluate_polynomial(coeffs, x)
            print(f"   Verify x={x}: polynomial value={poly_val}, expected={y} {' ' if poly_val == y else ' '}")
        
        return True
        
    except Exception as e:
        print(f"\n  Advanced operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finite_field_crypto():
    
    try:
        # Use a secure prime
        ff = FiniteField(bit_length=128)
        
        print("1. Testing random element generation...")
        
        # Generate random elements
        rand_elem = ff.random_element()
        rand_nonzero = ff.random_nonzero()
        
        print(f"   Random element: {rand_elem} (range: 0 to {ff.p-1})")
        print(f"   Random nonzero element: {rand_nonzero} (nonzero: {' ' if rand_nonzero != 0 else ' '})")
        
        print("\n2. Testing hash-to-field...")
        
        test_data = b"DeCart Web3.0 AI Queries"
        hashed_elem = ff.hash_to_field(test_data)
        print(f"   Data: {test_data[:20]}...")
        print(f"   Hash-to-field element: {hashed_elem}")
        
        # Verify it is in the field
        print(f"   In-field verification: 0 <= {hashed_elem} < {ff.p} {' ' if 0 <= hashed_elem < ff.p else ' '}")
        
        print("\n3. Testing byte conversion...")
        
        # Convert to bytes and recover
        original = 123456789
        elem_bytes = ff.to_bytes(original)
        recovered = ff.from_bytes(elem_bytes)
        
        print(f"   Original: {original}")
        print(f"   Byte length: {len(elem_bytes)}")
        print(f"   Recovered: {recovered}")
        print(f"   Consistency: {'' if original % ff.p == recovered else ' '}")
        
        print("\n4. Testing Legendre symbol...")
        
        test_values = [2, 3, 5, 7, 11]
        for val in test_values:
            legendre = ff.legendre_symbol(val)
            is_residue = ff.is_quadratic_residue(val)
            print(f"   Legendre symbol({val}/{ff.p}) = {legendre}, quadratic residue: {is_residue}")
        
        return True
        
    except Exception as e:
        print(f"\n  Cryptographic operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
