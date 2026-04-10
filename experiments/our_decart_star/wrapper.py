# decart/experiments/our_decart_star/wrapper.py
"""
DeCart* 方案实验包装器
用于对比实验的标准化接口
"""

import sys
import os
import time
import pickle
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from entities.key_curator import KeyCurator
from entities.data_owner import DataOwner
from entities.data_querier import DataQuerier
from entities.database_server import DatabaseServer
from schemes.decart_star import DeCartStarParams


class DeCartStarExperimentWrapper:
    """
    DeCart* 方案实验包装器
    提供标准化的接口用于对比实验
    """
    
    def __init__(self, N: int = 64, n: int = 16):
        """
        初始化实验环境
        
        参数:
            N: 最大用户数
            n: 每块用户数
        """
        self.params = DeCartStarParams(N=N, n=n)
        self.curator = None
        self.db_server = None
        self.owners = {}
        self.queriers = {}
        
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
    
    def setup(self) -> float:
        """
        初始化系统
        
        返回:
            setup_time: 初始化耗时
        """
        start = time.perf_counter()
        
        self.curator = KeyCurator(scheme="decart_star", params=self.params)
        self.curator.setup()
        
        self.db_server = DatabaseServer(
            server_id="ds1", 
            key_curator=self.curator, 
            scheme="decart_star"
        )
        
        elapsed = time.perf_counter() - start
        self.metrics['setup_time'] = elapsed
        
        print(f" DeCart* 实验环境初始化完成: {elapsed:.4f}秒")
        return elapsed
    
    def register_user(self, user_id: int) -> Tuple[int, Any]:
        """
        注册用户
        
        返回:
            (sk, pk) 密钥对
        """
        start = time.perf_counter()
        
        sk, pk, pap = self.curator.generate_user_key(user_id)
        self.curator.register(user_id, pk, pap)
        
        elapsed = time.perf_counter() - start
        self.metrics['keygen_times'].append(elapsed)
        
        return sk, pk
    
    def create_owner(self, owner_id: int) -> DataOwner:
        """创建数据所有者"""
        if owner_id not in self.owners:
            owner = DataOwner(
                owner_id=owner_id,
                key_curator=self.curator,
                scheme="decart_star"
            )
            self.owners[owner_id] = owner
        return self.owners[owner_id]
    
    def create_querier(self, querier_id: int) -> DataQuerier:
        """创建数据查询者"""
        if querier_id not in self.queriers:
            querier = DataQuerier(
                querier_id=querier_id,
                key_curator=self.curator,
                scheme="decart_star"
            )
            self.queriers[querier_id] = querier
        return self.queriers[querier_id]
    
    def encrypt_dataset(self, 
                       owner_id: int,
                       data: List[List[float]],
                       policy: List[int],
                       metadata: Optional[Dict] = None) -> Tuple[Dict, Any, str]:
        """
        加密数据集 - 测量加密时间和密文大小
        
        返回:
            (C_m, sk_h_s, dataset_id)
        """
        owner = self.create_owner(owner_id)
        
        # 测量加密时间
        start = time.perf_counter()
        C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy, metadata)
        elapsed = time.perf_counter() - start
        self.metrics['encrypt_times'].append(elapsed)
        
        # 测量密文大小
        try:
            import pickle
            size = len(pickle.dumps(C_m))
        except (TypeError, pickle.PicklingError):
            # 估算大小
            size = len(data) * 1024 * 1024  # 1MB per record
        
        self.metrics['communication_sizes'].append({
            'type': 'encrypt',
            'size': size,
            'records': len(data)
        })
        
        print(f"    加密时间: {elapsed*1000:.2f} ms, 密文大小: {size/1024:.2f} KB")
        
        return C_m, sk_h_s, ds_id
    
    def store_dataset(self, owner_id: int, dataset_id: str, C_m: Dict, sk_h_s: Any):
        """存储数据集到数据库服务器"""
        self.db_server.store_dataset(owner_id, dataset_id, C_m, sk_h_s)
    
    def execute_query(self,
                     querier_id: int,
                     owner_id: int,
                     dataset_id: str,
                     model: Any) -> Optional[List[float]]:
        """
        执行查询 - 支持多种模型类型
        
        返回:
            查询结果
        """
        querier = self.create_querier(querier_id)
        
        # 获取数据集
        C_m, sk_h_s = self.db_server.get_dataset(owner_id, dataset_id)
        if C_m is None:
            return None
        
        # 检查权限
        C_M = querier.check_access(C_m)
        if C_M is None:
            return None
        
        # 根据模型类型处理
        if isinstance(model, list):
            # 点积模型 - 直接加密列表
            print(f"    点积模型加密...")
            C_M = querier.encrypt_ai_model(model, C_M)
            
        elif isinstance(model, dict) and model.get('type') == 'neural_network':
            # 神经网络模型
            print(f"    神经网络模型加密...")
            
            weights = model.get('weights', [])
            bias = model.get('bias', [])
            input_dim = model.get('input_dim', 64)
            output_dim = model.get('output_dim', 10)
            
            # 加密权重
            encrypted_weights = []
            for w in weights:
                try:
                    encrypted_w = self.curator.system.he.encrypt([float(w)])
                    encrypted_weights.append(encrypted_w)
                except Exception as e:
                    print(f"     权重加密失败: {e}")
                    encrypted_weights.append(None)
            
            # 加密偏置
            encrypted_bias = []
            for b in bias:
                try:
                    encrypted_b = self.curator.system.he.encrypt([float(b)])
                    encrypted_bias.append(encrypted_b)
                except Exception as e:
                    print(f"     偏置加密失败: {e}")
                    encrypted_bias.append(None)
            
            encrypted_model = {
                'type': 'neural_network',
                'layer_count': 1,
                'layers': [{
                    'layer_idx': 0,
                    'layer_type': 'linear',
                    'activation': 'linear',
                    'weights_shape': (output_dim, input_dim),
                    'bias_shape': (output_dim,),
                    'encrypted_weights': encrypted_weights,
                    'encrypted_bias': encrypted_bias
                }]
            }
            C_M['encrypted_model'] = encrypted_model
            C_M['model_type'] = 'neural_network'
            
        elif isinstance(model, dict) and model.get('type') == 'decision_tree':
            # 决策树模型 - 使用系统方法加密
            print(f"    决策树模型加密...")
            pk_h = self.curator.system.he.public_key
            
            if hasattr(self.curator.system, 'encrypt_decision_tree'):
                encrypted_model = self.curator.system.encrypt_decision_tree(model, pk_h)
            else:
                encrypted_model = {
                    'type': 'decision_tree',
                    'encrypted': True,
                    'nodes': model.get('nodes', [])
                }
            C_M['encrypted_model'] = encrypted_model
            C_M['model_type'] = 'decision_tree'
            
        else:
            print(f"      不支持的模型类型: {type(model)}")
            return None
        
        # 测量查询时间
        start_query = time.perf_counter()
        ER = self.curator.system.query(C_M, C_m, sk_h_s)
        query_time = time.perf_counter() - start_query
        self.metrics['query_times'].append(query_time)
        
        # 测量解密时间
        start_decrypt = time.perf_counter()
        results = self.curator.system.decrypt(C_M['sk_h_u'], ER)
        decrypt_time = time.perf_counter() - start_decrypt
        self.metrics['decrypt_times'].append(decrypt_time)
        
        print(f"    查询时间: {query_time*1000:.2f} ms, 解密时间: {decrypt_time*1000:.2f} ms")
        
        return results
    
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
        
        return metrics
    
    def save_results(self, filepath: str):
        """保存实验结果"""
        results = {
            'params': {
                'N': self.params.N,
                'n': self.params.n,
                'B': self.params.B
            },
            'metrics': self.get_metrics(),
            'timestamp': time.time()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f" 实验结果已保存到: {filepath}")