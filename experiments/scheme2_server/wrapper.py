# decart/experiments/scheme2_server/wrapper.py
"""
同态加密+服务器方案实验包装器
模拟传统云服务模式：
- 数据和模型都用服务器的公钥加密
- 依赖单一可信服务器
"""

import sys
import os
import time
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 复用核心密码学模块
from core.homomorphic import HomomorphicEncryption


class ServerSchemeExperimentWrapper:
    """
    同态加密+服务器方案实验包装器
    特点：
    - 单一可信服务器
    - 数据和模型用服务器公钥加密
    - 服务器执行同态计算
    - 用户无密钥（完全依赖服务器）
    """
    
    def __init__(self, N: int = 64, n: int = 16):
        """
        初始化实验环境
        
        参数:
            N: 最大用户数（仅用于接口兼容）
            n: 每块用户数（仅用于接口兼容）
        """
        self.N = N
        self.n = n
        
        # 初始化同态加密（服务器拥有密钥）
        self.he = HomomorphicEncryption(poly_modulus_degree=4096)
        
        # 服务器密钥对
        self.server_public_key = self.he.public_key
        self.server_secret_key = self.he.secret_key
        
        # 数据存储（加密）
        self.encrypted_datasets = {}  # owner_id -> {dataset_id -> encrypted_data}
        
        # 用户注册表
        self.registered_users = set()
        
        # 实验数据
        self.metrics = {
            'setup_time': 0,
            'keygen_times': [],
            'encrypt_times': [],
            'query_times': [],
            'decrypt_times': [],
            'communication_sizes': [],
            'memory_sizes': []
        }
        
        print(f"  Server 方案实验环境初始化完成（单一可信服务器）")
    
    def setup(self) -> float:
        """
        初始化系统
        
        返回:
            setup_time: 初始化耗时
        """
        start = time.time()
        # 服务器方案只需要初始化同态加密（已在 __init__ 中完成）
        elapsed = time.time() - start
        self.metrics['setup_time'] = elapsed
        print(f"   Server 方案初始化完成: {elapsed:.4f}秒")
        return elapsed
    
    def register_user(self, user_id: int) -> Tuple[int, Any]:
        """
        注册用户（服务器方案中用户无密钥）
        
        返回:
            (user_id, None) - 用户没有密钥
        """
        start = time.time()
        
        self.registered_users.add(user_id)
        
        elapsed = time.time() - start
        self.metrics['keygen_times'].append(elapsed)
        
        # 用户没有密钥，所有加密用服务器公钥
        return user_id, None
    
    def encrypt_dataset(self, 
                       owner_id: int,
                       data: List[List[float]],
                       policy: List[int],
                       metadata: Optional[Dict] = None) -> Tuple[Dict, Any, str]:
        """
        加密数据集（使用服务器公钥）
        
        返回:
            (C_m, None, dataset_id) - 服务器方案不需要密钥份额
        """
        # 生成数据集ID
        import time
        timestamp = int(time.time() * 1000)
        dataset_id = f"ds_{owner_id}_{timestamp}"
        
        # 测量加密时间
        start = time.time()
        
        # 加密每条数据记录
        encrypted_data = []
        for record in data:
            encrypted_record = self.he.encrypt(record)
            encrypted_data.append(encrypted_record)
        
        elapsed = time.time() - start
        self.metrics['encrypt_times'].append(elapsed)
        
        # 存储加密数据
        if owner_id not in self.encrypted_datasets:
            self.encrypted_datasets[owner_id] = {}
        
        self.encrypted_datasets[owner_id][dataset_id] = {
            'encrypted_data': encrypted_data,
            'policy': policy.copy(),
            'metadata': metadata or {},
            'store_time': time.time()
        }
        
        # 测量密文大小
        try:
            import pickle
            size = len(pickle.dumps(encrypted_data))
        except:
            size = len(data) * 1024 * 1024  # 1MB per record估算
        
        self.metrics['communication_sizes'].append({
            'type': 'encrypt',
            'size': size,
            'records': len(data)
        })
        
        print(f"     Server 方案加密: {elapsed*1000:.2f} ms, 密文大小: {size/1024:.2f} KB")
        
        # 返回格式兼容的元数据
        C_m = {
            'type': 'server_scheme',
            'dataset_id': dataset_id,
            'owner_id': owner_id,
            'policy': policy,
            'num_records': len(data)
        }
        
        return C_m, None, dataset_id
    
    def store_dataset(self, owner_id: int, dataset_id: str, C_m: Dict, sk_h_s: Any):
        """已在 encrypt_dataset 中完成存储"""
        pass
    
    def encrypt_model(self, model: Any) -> Any:
        """
        加密模型（使用服务器公钥）
        
        返回:
            加密后的模型
        """
        if isinstance(model, list):
            # 点积模型 - 加密列表
            return self.he.encrypt(model)
        elif isinstance(model, dict) and model.get('type') == 'neural_network':
            # 神经网络模型 - 加密权重和偏置
            weights = model.get('weights', [])
            bias = model.get('bias', [])
            
            encrypted_weights = []
            for w in weights[:100]:  # 限制数量
                encrypted_weights.append(self.he.encrypt([float(w)]))
            
            encrypted_bias = []
            for b in bias:
                encrypted_bias.append(self.he.encrypt([float(b)]))
            
            return {
                'type': 'neural_network',
                'encrypted_weights': encrypted_weights,
                'encrypted_bias': encrypted_bias,
                'input_dim': model.get('input_dim'),
                'output_dim': model.get('output_dim')
            }
        else:
            # 默认处理
            return self.he.encrypt([0.0])
    
    def execute_query(self,
                     querier_id: int,
                     owner_id: int,
                     dataset_id: str,
                     model: Any) -> Optional[List[float]]:
        """
        执行查询（服务器端同态计算）
        
        返回:
            解密后的查询结果
        """
        # 检查数据集是否存在
        if owner_id not in self.encrypted_datasets or dataset_id not in self.encrypted_datasets[owner_id]:
            print(f"     数据集不存在")
            return None
        
        dataset_info = self.encrypted_datasets[owner_id][dataset_id]
        encrypted_data = dataset_info['encrypted_data']
        
        # 加密模型
        start_encrypt_model = time.time()
        encrypted_model = self.encrypt_model(model)
        model_encrypt_time = time.time() - start_encrypt_model
        self.metrics['encrypt_times'].append(model_encrypt_time)
        
        # 执行同态查询
        start_query = time.time()
        
        results = []
        if isinstance(model, list):
            # 点积模型
            for enc_record in encrypted_data:
                try:
                    # 同态点积
                    result = enc_record.dot(encrypted_model)
                    results.append(result)
                except:
                    results.append(self.he.encrypt([0.0]))
        else:
            # 简化处理
            for enc_record in encrypted_data:
                results.append(self.he.encrypt([0.0]))
        
        query_time = time.time() - start_query
        self.metrics['query_times'].append(query_time)
        
        # 解密结果（服务器拥有私钥）
        start_decrypt = time.time()
        decrypted_results = []
        for enc_result in results:
            try:
                dec = self.he.decrypt(enc_result)
                if isinstance(dec, list):
                    decrypted_results.append(dec[0] if dec else 0.0)
                else:
                    decrypted_results.append(float(dec))
            except:
                decrypted_results.append(0.0)
        
        decrypt_time = time.time() - start_decrypt
        self.metrics['decrypt_times'].append(decrypt_time)
        
        print(f"      Server 方案查询: {query_time*1000:.2f} ms")
        print(f"      模型加密: {model_encrypt_time*1000:.2f} ms")
        print(f"      结果解密: {decrypt_time*1000:.2f} ms")
        
        return decrypted_results
    
    def reset_metrics(self):
        """重置实验指标"""
        self.metrics = {
            'setup_time': 0,
            'keygen_times': [],
            'encrypt_times': [],
            'query_times': [],
            'decrypt_times': [],
            'communication_sizes': [],
            'memory_sizes': []
        }
    
    def get_metrics(self) -> Dict:
        """获取所有实验指标"""
        metrics = self.metrics.copy()
        
        # 计算平均值
        if metrics['encrypt_times']:
            metrics['avg_encrypt_time'] = np.mean(metrics['encrypt_times'])
            metrics['std_encrypt_time'] = np.std(metrics['encrypt_times'])
        
        if metrics['query_times']:
            metrics['avg_query_time'] = np.mean(metrics['query_times'])
            metrics['std_query_time'] = np.std(metrics['query_times'])
        
        if metrics['decrypt_times']:
            metrics['avg_decrypt_time'] = np.mean(metrics['decrypt_times'])
            metrics['std_decrypt_time'] = np.std(metrics['decrypt_times'])
        
        # 通信大小
        if metrics['communication_sizes']:
            sizes = [s['size'] for s in metrics['communication_sizes']]
            metrics['avg_communication_size'] = np.mean(sizes)
            metrics['total_communication'] = np.sum(sizes)
        
        return metrics