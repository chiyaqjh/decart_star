# decart/experiments/our_decart/owner.py
"""
DeCart 方案数据所有者实验接口
用于对比实验
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

from entities.data_owner import DataOwner as BaseDataOwner


class DataOwner(BaseDataOwner):
    """
    DeCart 方案数据所有者（实验版）
    添加实验测量功能
    """
    
    def __init__(self, owner_id: int, key_curator, scheme: str = "decart"):
        super().__init__(owner_id, key_curator, scheme)
        
        # 实验测量数据
        self.experiment_metrics = {
            'encrypt_times': [],
            'policy_sizes': [],
            'data_sizes': [],
            'model_encrypt_times': [],
            'communication_overhead': []
        }
    
    def encrypt_data_with_metrics(self,
                                 data_records: List[List[float]],
                                 access_policy: List[int],
                                 metadata: Optional[Dict] = None,
                                 store_original: bool = False) -> Tuple[Dict, Any, str, Dict]:
        """
        加密数据并测量指标
        
        返回:
            (C_m, sk_h_s, dataset_id, metrics)
        """
        # 记录输入大小
        data_size = sum(len(record) for record in data_records) * 8  # float 8字节
        policy_size = len(access_policy) * 4  # int 4字节
        
        # 测量加密时间
        start = time.perf_counter()
        C_m, sk_h_s, ds_id = self.encrypt_data(
            data_records, access_policy, metadata, store_original=store_original
        )
        encrypt_time = time.perf_counter() - start
        
        # 测量密文大小
        import sys
        import pickle
        cipher_size = sys.getsizeof(pickle.dumps(C_m))
        
        # 记录指标
        metrics = {
            'encrypt_time': encrypt_time,
            'data_size': data_size,
            'policy_size': policy_size,
            'cipher_size': cipher_size,
            'num_records': len(data_records),
            'record_dim': len(data_records[0]) if data_records else 0,
            'policy_length': len(access_policy)
        }
        
        self.experiment_metrics['encrypt_times'].append(encrypt_time)
        self.experiment_metrics['policy_sizes'].append(policy_size)
        self.experiment_metrics['data_sizes'].append(data_size)
        self.experiment_metrics['communication_overhead'].append(cipher_size)
        
        print(f"\n [DataOwner {self.owner_id}] 加密指标:")
        print(f"   加密时间: {encrypt_time*1000:.2f} ms")
        print(f"   数据大小: {data_size/1024:.2f} KB")
        print(f"   密文大小: {cipher_size/1024:.2f} KB")
        print(f"   膨胀率: {cipher_size/data_size:.2f}x")
        
        return C_m, sk_h_s, ds_id, metrics
    
    def encrypt_model_with_metrics(self,
                                  model_id: str,
                                  access_policy: List[int]) -> Tuple[Dict, str, Dict]:
        """
        加密模型并测量指标
        
        返回:
            (encrypted_model, encrypted_model_id, metrics)
        """
        # 获取模型信息
        model_info = self.model_metadata.get(model_id, {})
        architecture = model_info.get('architecture', {})
        
        # 计算模型大小
        if 'weights' in architecture:
            model_size = len(architecture['weights']) * 8  # float 8字节
        elif 'combined_weights' in architecture:
            model_size = len(architecture['combined_weights']) * 8
        else:
            model_size = 0
        
        # 测量加密时间
        start = time.perf_counter()
        encrypted_model, enc_id = self.encrypt_model(model_id, access_policy)
        encrypt_time = time.perf_counter() - start
        
        # 测量密文大小
        import sys
        import pickle
        cipher_size = sys.getsizeof(pickle.dumps(encrypted_model))
        
        metrics = {
            'encrypt_time': encrypt_time,
            'model_size': model_size,
            'cipher_size': cipher_size,
            'policy_length': len(access_policy),
            'model_type': model_info.get('model_type', 'unknown'),
            'encrypt_time_ms': encrypt_time * 1000
        }
        
        self.experiment_metrics['model_encrypt_times'].append(encrypt_time)
        
        print(f"\n [DataOwner {self.owner_id}] 模型加密指标:")
        print(f"   加密时间: {encrypt_time*1000:.2f} ms")
        print(f"   模型大小: {model_size/1024:.2f} KB")
        print(f"   密文大小: {cipher_size/1024:.2f} KB")
        print(f"   膨胀率: {cipher_size/model_size:.2f}x")
        
        return encrypted_model, enc_id, metrics
    
    def get_experiment_metrics(self) -> Dict:
        """获取实验指标统计"""
        metrics = {}
        
        if self.experiment_metrics['encrypt_times']:
            times = self.experiment_metrics['encrypt_times']
            metrics['avg_encrypt_time'] = np.mean(times)
            metrics['std_encrypt_time'] = np.std(times)
            metrics['min_encrypt_time'] = np.min(times)
            metrics['max_encrypt_time'] = np.max(times)
        
        if self.experiment_metrics['model_encrypt_times']:
            times = self.experiment_metrics['model_encrypt_times']
            metrics['avg_model_encrypt_time'] = np.mean(times)
            metrics['std_model_encrypt_time'] = np.std(times)
        
        if self.experiment_metrics['communication_overhead']:
            sizes = self.experiment_metrics['communication_overhead']
            metrics['avg_cipher_size'] = np.mean(sizes)
            metrics['total_communication'] = sum(sizes)
        
        if self.experiment_metrics['data_sizes'] and self.experiment_metrics['communication_overhead']:
            ratios = [c/d for c, d in zip(self.experiment_metrics['communication_overhead'], 
                                          self.experiment_metrics['data_sizes'])]
            metrics['avg_expansion_ratio'] = np.mean(ratios)
        
        return metrics
    
    def reset_metrics(self):
        """重置实验指标"""
        self.experiment_metrics = {
            'encrypt_times': [],
            'policy_sizes': [],
            'data_sizes': [],
            'model_encrypt_times': [],
            'communication_overhead': []
        }