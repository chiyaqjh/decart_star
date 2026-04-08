# decart/config.py
"""
DeCart 配置文件
定义系统参数和常量
"""

import os
from typing import Optional

class Config:
    """配置类 - 存储所有系统参数"""
    
    # ========== 系统安全参数 ==========
    SECURITY_PARAMETER: int = 256  # λ - 安全参数
    
    # ========== 双线性配对参数 ==========
    CURVE_TYPE: str = "bls12_381"  # 使用的椭圆曲线类型
    
    # ========== 系统容量参数 ==========
    MAX_USERS: int = 1000      # N: 系统支持的最大用户数
    BLOCK_SIZE: int = 10       # n: 每个块中的用户数
    NUM_BLOCKS: Optional[int] = None  # B: 总块数，将在初始化时计算
    
    # ========== 同态加密参数 ==========
    HOMOMORPHIC_SCHEME: str = "CKKS"  # 使用的同态加密方案
    
    # CKKS 特定参数
    POLY_MODULUS_DEGREE: int = 8192     # 多项式模数阶
    COEFF_MOD_BIT_SIZES: list = [60, 40, 40, 60]  # 系数模数位大小
    SCALE: int = 2**40                   # 缩放因子
    
    # ========== 数据库配置 ==========
    DATABASE_SERVERS: int = 3           # 数据库服务器数量
    THRESHOLD_SERVERS: int = 2          # 解密所需的最小服务器数
    
    # ========== 哈希函数配置 ==========
    HASH_ALGORITHM: str = "sha256"      # 使用的哈希算法
    HASH_OUTPUT_SIZE: int = 32          # 哈希输出大小（字节）
    
    # ========== 文件路径配置 ==========
    DATA_DIR: str = "data"              # 数据存储目录
    KEYS_DIR: str = "keys"              # 密钥存储目录
    LOGS_DIR: str = "logs"              # 日志存储目录
    
    # ========== 性能参数 ==========
    USE_PARALLEL: bool = False          # 是否使用并行计算
    BATCH_SIZE: int = 100               # 批处理大小
    
    @classmethod
    def initialize(cls):
        """初始化配置参数"""
        print("初始化DeCart配置...")
        
        # 计算块数 B = ceil(N / n)
        cls.NUM_BLOCKS = (cls.MAX_USERS + cls.BLOCK_SIZE - 1) // cls.BLOCK_SIZE
        
        # 创建必要的目录
        dirs_to_create = [cls.DATA_DIR, cls.KEYS_DIR, cls.LOGS_DIR]
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")
        
        # 打印配置信息
        cls.print_config()
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("\n" + "="*50)
        print("DeCart 系统配置")
        print("="*50)
        print(f"安全参数 (λ): {cls.SECURITY_PARAMETER}")
        print(f"最大用户数 (N): {cls.MAX_USERS}")
        print(f"块大小 (n): {cls.BLOCK_SIZE}")
        print(f"总块数 (B): {cls.NUM_BLOCKS}")
        print(f"双线性曲线: {cls.CURVE_TYPE}")
        print(f"同态加密方案: {cls.HOMOMORPHIC_SCHEME}")
        print(f"数据库服务器数: {cls.DATABASE_SERVERS}")
        print(f"解密阈值: {cls.THRESHOLD_SERVERS}")
        print("="*50)
    
    @classmethod
    def update_parameters(cls, 
                         max_users: Optional[int] = None,
                         block_size: Optional[int] = None,
                         security_param: Optional[int] = None):
        """更新系统参数"""
        if max_users is not None:
            cls.MAX_USERS = max_users
        
        if block_size is not None:
            cls.BLOCK_SIZE = block_size
        
        if security_param is not None:
            cls.SECURITY_PARAMETER = security_param
        
        # 重新计算块数
        cls.NUM_BLOCKS = (cls.MAX_USERS + cls.BLOCK_SIZE - 1) // cls.BLOCK_SIZE
        
        print("参数已更新")
        cls.print_config()
    
    @classmethod
    def get_prime_order(cls) -> int:
        """获取有限域的素数阶 p（模拟值，实际应从双线性配对库获取）"""
        # 这里返回一个大的素数用于测试
        # 在实际使用中，应从双线性配对库获取真实的 p
        import sympy
        return sympy.randprime(2**254, 2**256 - 1)
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置的有效性"""
        try:
            # 验证基本参数
            assert cls.SECURITY_PARAMETER >= 128, "安全参数至少128位"
            assert cls.MAX_USERS > 0, "最大用户数必须为正"
            assert cls.BLOCK_SIZE > 0, "块大小必须为正"
            assert cls.BLOCK_SIZE <= cls.MAX_USERS, "块大小不能超过最大用户数"
            assert cls.DATABASE_SERVERS > 0, "至少需要1个数据库服务器"
            assert cls.THRESHOLD_SERVERS > 0, "阈值必须为正"
            assert cls.THRESHOLD_SERVERS <= cls.DATABASE_SERVERS, "阈值不能超过服务器数"
            
            # 验证同态加密参数
            if cls.HOMOMORPHIC_SCHEME == "CKKS":
                valid_poly_modulus = [1024, 2048, 4096, 8192, 16384, 32768]
                assert cls.POLY_MODULUS_DEGREE in valid_poly_modulus, f"多项式阶必须是{valid_poly_modulus}之一"
                assert cls.SCALE > 0, "缩放因子必须为正"
            
            print("✓ 配置验证通过")
            return True
            
        except AssertionError as e:
            print(f"✗ 配置验证失败: {e}")
            return False


# 测试函数
def test_config():
    """测试配置模块"""
    print("测试配置模块...")
    
    # 创建配置实例
    config = Config
    
    # 测试初始化
    config.initialize()
    
    # 测试参数更新
    config.update_parameters(max_users=500, block_size=20)
    
    # 测试验证
    is_valid = config.validate_config()
    
    # 测试获取素数阶（模拟）
    prime_order = config.get_prime_order()
    print(f"生成的素数阶 (p): {prime_order}")
    print(f"素数位数: {prime_order.bit_length()}")
    
    # 测试目录创建
    import os
    for directory in [config.DATA_DIR, config.KEYS_DIR, config.LOGS_DIR]:
        assert os.path.exists(directory), f"目录 {directory} 不存在"
        print(f"✓ 目录存在: {directory}")
    
    print("\n✓ 配置模块测试完成")
    return True


if __name__ == "__main__":
    # 运行测试
    success = test_config()
    if success:
        print("\n✅ 配置模块实现完成，可进行测试")
    else:
        print("\n❌ 配置模块测试失败")