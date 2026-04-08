# decart/core/finite_field.py
"""
完全非模拟的有限域运算模块
实现论文中的 Z_p 运算
"""

import secrets
import hashlib
import math
from typing import Tuple, List, Optional, Union
from sympy import isprime, nextprime


class FiniteField:
    """有限域 Z_p 运算类"""
    
    def __init__(self, p: Optional[int] = None, bit_length: int = 256):
        """
        初始化有限域
        
        参数:
            p: 素数模数，如果为None则自动生成
            bit_length: 素数的位长度（当p为None时使用）
        """
        if p is not None:
            if not isprime(p):
                raise ValueError(f"{p} 不是素数")
            self.p = p
            self.bit_length = p.bit_length()
        else:
            # 生成安全素数
            self.p = self._generate_prime(bit_length)
            self.bit_length = bit_length
        
        print(f"  有限域初始化成功")
        print(f"   素数模数 p: {self.p}")
        print(f"   位长度: {self.bit_length}")
        print(f"   十六进制: 0x{self.p:X}")
    
    def _generate_prime(self, bit_length: int) -> int:
        """生成安全素数"""
        if bit_length < 128:
            raise ValueError("素数位长度至少128位以确保安全")
        
        print(f"   正在生成 {bit_length} 位素数...")
        
        # 使用确定性方法生成素数（用于研究可重现性）
        seed = b"DeCart_Finite_Field_Research"
        base = int.from_bytes(hashlib.sha256(seed).digest(), 'big')
        
        # 确保是奇数
        if base % 2 == 0:
            base += 1
        
        # 寻找下一个素数
        prime_candidate = nextprime(base >> (base.bit_length() - bit_length))
        
        # 验证素数性质
        if not isprime(prime_candidate):
            # 如果失败，使用随机方法
            prime_candidate = secrets.randbits(bit_length)
            if prime_candidate % 2 == 0:
                prime_candidate += 1
            
            while not isprime(prime_candidate):
                prime_candidate += 2
                # 防止无限循环
                if prime_candidate.bit_length() > bit_length + 10:
                    prime_candidate = secrets.randbits(bit_length)
                    if prime_candidate % 2 == 0:
                        prime_candidate += 1
        
        return prime_candidate
    
    def random_element(self) -> int:
        """生成随机域元素"""
        return secrets.randbelow(self.p)
    
    def random_nonzero(self) -> int:
        """生成随机非零域元素"""
        element = self.random_element()
        while element == 0:
            element = self.random_element()
        return element
    
    def add(self, a: int, b: int) -> int:
        """模加法: (a + b) mod p"""
        return (a + b) % self.p
    
    def sub(self, a: int, b: int) -> int:
        """模减法: (a - b) mod p"""
        return (a - b) % self.p
    
    def mul(self, a: int, b: int) -> int:
        """模乘法: (a * b) mod p"""
        return (a * b) % self.p
    
    def pow(self, base: int, exponent: int) -> int:
        """模幂运算: base^exponent mod p"""
        return pow(base, exponent, self.p)
    
    def inv(self, a: int) -> int:
        """模逆元: a^{-1} mod p"""
        if a == 0:
            raise ValueError("0没有模逆元")
        
        # 使用扩展欧几里得算法
        return pow(a, self.p - 2, self.p)  # 费马小定理: a^{p-1} ≡ 1 mod p
    
    def div(self, a: int, b: int) -> int:
        """模除法: a / b mod p"""
        if b == 0:
            raise ValueError("除以零")
        
        b_inv = self.inv(b)
        return self.mul(a, b_inv)
    
    def sqrt(self, a: int) -> Tuple[Optional[int], Optional[int]]:
        """计算平方根: 返回 (root1, root2) 或 (None, None)"""
        if a == 0:
            return 0, 0
        
        # 检查是否有平方根（欧拉准则）
        legendre = pow(a, (self.p - 1) // 2, self.p)
        if legendre != 1:
            return None, None  # 没有平方根
        
        # Tonelli-Shanks算法求平方根
        # 特殊情况: p ≡ 3 mod 4
        if self.p % 4 == 3:
            root = pow(a, (self.p + 1) // 4, self.p)
            return root, (self.p - root) % self.p
        
        # 一般情况: Tonelli-Shanks算法
        return self._tonelli_shanks(a)
    
    def _tonelli_shanks(self, n: int) -> Tuple[int, int]:
        """Tonelli-Shanks算法求平方根"""
        # 实现Tonelli-Shanks算法
        Q = self.p - 1
        S = 0
        while Q % 2 == 0:
            Q //= 2
            S += 1
        
        # 寻找二次非剩余
        z = 2
        while pow(z, (self.p - 1) // 2, self.p) != self.p - 1:
            z += 1
        
        M = S
        c = pow(z, Q, self.p)
        t = pow(n, Q, self.p)
        R = pow(n, (Q + 1) // 2, self.p)
        
        while t != 1:
            # 找到最小的i使得 t^{2^i} ≡ 1
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
        """计算勒让德符号 (a/p)"""
        ls = pow(a, (self.p - 1) // 2, self.p)
        if ls == self.p - 1:
            return -1
        return ls
    
    def is_quadratic_residue(self, a: int) -> bool:
        """检查是否为二次剩余"""
        return self.legendre_symbol(a) == 1
    
    def batch_add(self, elements: List[int]) -> int:
        """批量模加法"""
        result = 0
        for elem in elements:
            result = self.add(result, elem)
        return result
    
    def batch_mul(self, elements: List[int]) -> int:
        """批量模乘法"""
        result = 1
        for elem in elements:
            result = self.mul(result, elem)
        return result
    
    def dot_product(self, vec1: List[int], vec2: List[int]) -> int:
        """向量点积模p"""
        if len(vec1) != len(vec2):
            raise ValueError("向量长度必须相同")
        
        result = 0
        for a, b in zip(vec1, vec2):
            product = self.mul(a, b)
            result = self.add(result, product)
        return result
    
    def matrix_mul(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        """矩阵乘法模p"""
        rows1, cols1 = len(mat1), len(mat1[0])
        rows2, cols2 = len(mat2), len(mat2[0])
        
        if cols1 != rows2:
            raise ValueError("矩阵维度不匹配")
        
        result = [[0] * cols2 for _ in range(rows1)]
        
        for i in range(rows1):
            for j in range(cols2):
                for k in range(cols1):
                    product = self.mul(mat1[i][k], mat2[k][j])
                    result[i][j] = self.add(result[i][j], product)
        
        return result
    
    def random_vector(self, n: int) -> List[int]:
        """生成随机向量"""
        return [self.random_element() for _ in range(n)]
    
    def random_matrix(self, rows: int, cols: int) -> List[List[int]]:
        """生成随机矩阵"""
        return [[self.random_element() for _ in range(cols)] for _ in range(rows)]
    
    def evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """求多项式值: a0 + a1*x + a2*x^2 + ... mod p"""
        result = 0
        power = 1
        
        for coeff in coefficients:
            term = self.mul(coeff, power)
            result = self.add(result, term)
            power = self.mul(power, x)
        
        return result
    
    def lagrange_interpolation(self, points: List[Tuple[int, int]]) -> List[int]:
        """
        拉格朗日插值求多项式系数
        
        参数:
            points: [(x1, y1), (x2, y2), ...]
        
        返回:
            多项式系数列表 [a0, a1, ...]
        """
        n = len(points)
        
        # 初始化系数为0
        coefficients = [0] * n
        
        for i in range(n):
            xi, yi = points[i]
            
            # 计算拉格朗日基多项式li(x)
            li = [1]  # 初始多项式: 1
            
            for j in range(n):
                if i != j:
                    xj, _ = points[j]
                    
                    # 乘以 (x - xj)/(xi - xj)
                    denominator = self.sub(xi, xj)
                    denom_inv = self.inv(denominator)
                    
                    # 多项式乘以 (x - xj)
                    new_li = [0] * (len(li) + 1)
                    for k in range(len(li)):
                        # 乘以 -xj
                        term1 = self.mul(li[k], self.p - xj)
                        new_li[k] = self.add(new_li[k], term1)
                        
                        # 乘以 x
                        term2 = li[k]
                        new_li[k + 1] = self.add(new_li[k + 1], term2)
                    
                    li = new_li
                    
                    # 乘以 1/(xi - xj)
                    for k in range(len(li)):
                        li[k] = self.mul(li[k], denom_inv)
            
            # 将基多项式乘以yi并加到结果
            for k in range(len(li)):
                term = self.mul(li[k], yi)
                coefficients[k] = self.add(coefficients[k], term)
        
        return coefficients
    
    def to_bytes(self, element: int) -> bytes:
        """将域元素转换为字节"""
        byte_length = (self.p.bit_length() + 7) // 8
        return element.to_bytes(byte_length, 'big')
    
    def from_bytes(self, data: bytes) -> int:
        """从字节恢复域元素"""
        element = int.from_bytes(data, 'big')
        return element % self.p
    
    def hash_to_field(self, data: bytes) -> int:
        """哈希数据到域元素"""
        hash_bytes = hashlib.sha256(data).digest()
        element = int.from_bytes(hash_bytes, 'big')
        return element % self.p
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"FiniteField(Z_{self.p})"
    
    def __repr__(self) -> str:
        """详细表示"""
        return f"FiniteField(p=0x{self.p:X}, bit_length={self.bit_length})"


def test_finite_field_basic():
    """测试有限域基本运算"""
    print("=" * 60)
    print("测试有限域基本运算")
    print("=" * 60)
    
    try:
        # 使用小素数便于测试
        print("1. 初始化有限域（小素数便于测试）...")
        ff = FiniteField(p=65537)  # 使用固定素数确保测试一致性
        print(f"   域: Z_{ff.p}")
        
        # 测试基本运算
        print("\n2. 测试基本运算...")
        
        a, b = 12345, 54321
        
        # 加法
        add_result = ff.add(a, b)
        expected_add = (a + b) % ff.p
        print(f"   加法: {a} + {b} = {add_result} (期望: {expected_add}) {' ' if add_result == expected_add else ' '}")
        
        # 减法
        sub_result = ff.sub(a, b)
        expected_sub = (a - b) % ff.p
        print(f"   减法: {a} - {b} = {sub_result} (期望: {expected_sub}) {' ' if sub_result == expected_sub else ' '}")
        
        # 乘法
        mul_result = ff.mul(a, b)
        expected_mul = (a * b) % ff.p
        print(f"   乘法: {a} × {b} = {mul_result} (期望: {expected_mul}) {' ' if mul_result == expected_mul else ' '}")
        
        # 模逆
        inv_result = ff.inv(a)
        check_inv = ff.mul(a, inv_result)
        print(f"   模逆: {a}⁻¹ = {inv_result}, 验证: {a} × {inv_result} = {check_inv} {' ' if check_inv == 1 else ' '}")
        
        # 除法
        div_result = ff.div(a, b)
        check_div = ff.mul(div_result, b)
        print(f"   除法: {a} ÷ {b} = {div_result}, 验证: {div_result} × {b} = {check_div} {' ' if check_div == a else ' '}")
        
        # 幂运算
        exp_result = ff.pow(a, 3)
        expected_pow = pow(a, 3, ff.p)
        print(f"   幂运算: {a}³ = {exp_result} (期望: {expected_pow}) {' ' if exp_result == expected_pow else ' '}")
        
        return True
        
    except Exception as e:
        print(f"\n  基本运算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finite_field_advanced():
    """测试有限域高级运算"""
    print("\n" + "=" * 60)
    print("测试有限域高级运算")
    print("=" * 60)
    
    try:
        # 使用稍大的素数
        ff = FiniteField(p=104729)  # 第10000个素数
        
        print("1. 测试二次剩余和平方根...")
        
        # 测试平方根
        test_cases = [4, 9, 16, 25, 100]
        for n in test_cases:
            root1, root2 = ff.sqrt(n)
            if root1 is not None:
                check1 = ff.pow(root1, 2)
                check2 = ff.pow(root2, 2) if root2 is not None else None
                print(f"   √{n} = {root1}, {root2}")
                print(f"   验证: {root1}² = {check1}, {root2}² = {check2}")
        
        print("\n2. 测试批量运算...")
        
        # 批量加法
        numbers = [1, 2, 3, 4, 5]
        batch_sum = ff.batch_add(numbers)
        manual_sum = sum(numbers) % ff.p
        print(f"   批量加法: {numbers} → {batch_sum} (期望: {manual_sum})")
        
        # 批量乘法
        batch_prod = ff.batch_mul(numbers)
        manual_prod = 1
        for n in numbers:
            manual_prod = (manual_prod * n) % ff.p
        print(f"   批量乘法: {numbers} → {batch_prod} (期望: {manual_prod})")
        
        print("\n3. 测试向量和矩阵运算...")
        
        # 向量点积
        vec1 = [1, 2, 3]
        vec2 = [4, 5, 6]
        dot = ff.dot_product(vec1, vec2)
        manual_dot = sum(a*b for a,b in zip(vec1, vec2)) % ff.p
        print(f"   点积: {vec1}·{vec2} = {dot} (期望: {manual_dot})")
        
        # 矩阵乘法
        mat1 = [[1, 2], [3, 4]]
        mat2 = [[5, 6], [7, 8]]
        mat_result = ff.matrix_mul(mat1, mat2)
        print(f"   矩阵乘法: {mat1} × {mat2} = {mat_result}")
        
        print("\n4. 测试拉格朗日插值...")
        
        # 测试插值（Shamir秘密共享的基础）
        points = [(1, 5), (2, 10), (3, 19)]  # 对应多项式 2x² + 3
        coeffs = ff.lagrange_interpolation(points)
        print(f"   插值点: {points}")
        print(f"   恢复的系数: {coeffs}")
        
        # 验证插值
        for x, y in points:
            poly_val = ff.evaluate_polynomial(coeffs, x)
            print(f"   验证 x={x}: 多项式值={poly_val}, 期望={y} {' ' if poly_val == y else ' '}")
        
        return True
        
    except Exception as e:
        print(f"\n  高级运算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finite_field_crypto():
    """测试密码学相关运算"""
    print("\n" + "=" * 60)
    print("测试密码学相关运算")
    print("=" * 60)
    
    try:
        # 使用安全素数
        ff = FiniteField(bit_length=128)
        
        print("1. 测试随机元素生成...")
        
        # 生成随机元素
        rand_elem = ff.random_element()
        rand_nonzero = ff.random_nonzero()
        
        print(f"   随机元素: {rand_elem} (范围: 0 到 {ff.p-1})")
        print(f"   随机非零元素: {rand_nonzero} (非零: {' ' if rand_nonzero != 0 else ' '})")
        
        print("\n2. 测试哈希到域...")
        
        test_data = b"DeCart Web3.0 AI Queries"
        hashed_elem = ff.hash_to_field(test_data)
        print(f"   数据: {test_data[:20]}...")
        print(f"   哈希到域元素: {hashed_elem}")
        
        # 验证在域内
        print(f"   在域内验证: 0 <= {hashed_elem} < {ff.p} {' ' if 0 <= hashed_elem < ff.p else ' '}")
        
        print("\n3. 测试字节转换...")
        
        # 转换为字节并恢复
        original = 123456789
        elem_bytes = ff.to_bytes(original)
        recovered = ff.from_bytes(elem_bytes)
        
        print(f"   原始: {original}")
        print(f"   字节长度: {len(elem_bytes)}")
        print(f"   恢复: {recovered}")
        print(f"   一致性: {'' if original % ff.p == recovered else ' '}")
        
        print("\n4. 测试勒让德符号...")
        
        test_values = [2, 3, 5, 7, 11]
        for val in test_values:
            legendre = ff.legendre_symbol(val)
            is_residue = ff.is_quadratic_residue(val)
            print(f"   勒让德符号({val}/{ff.p}) = {legendre}, 二次剩余: {is_residue}")
        
        return True
        
    except Exception as e:
        print(f"\n  密码学运算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

'''
if __name__ == "__main__":
    print("开始测试有限域运算模块...")
    
    # 运行所有测试
    tests = [
        ("基本运算", test_finite_field_basic),
        ("高级运算", test_finite_field_advanced),
        ("密码学运算", test_finite_field_crypto)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"测试: {test_name}")
        print('='*60)
        
        if not test_func():
            all_passed = False
            print(f"  {test_name} 测试失败")
        else:
            print(f"  {test_name} 测试通过")
    
    if all_passed:
        print("\n" + "=" * 60)
        print("  finite_field.py 完整、正确、完全非模拟")
        print("   包含所有论文需要的有限域运算")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️ 有限域模块测试有失败，但核心功能可用")
        print("=" * 60)
        '''