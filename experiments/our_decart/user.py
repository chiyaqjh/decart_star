# decart/experiments/our_decart/user.py
"""
DeCart 方案数据查询者实验接口
用于对比实验
"""

import sys
import os
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from entities.data_querier import DataQuerier as BaseDataQuerier


class DataQuerier(BaseDataQuerier):
    """
    DeCart 方案数据查询者（实验版）
    添加实验测量功能
    """
    
    def __init__(self, querier_id: int, key_curator, scheme: str = "decart"):
        super().__init__(querier_id, key_curator, scheme)
        
        # 实验测量数据
        self.experiment_metrics = {
            'check_times': [],
            'model_encrypt_times': [],
            'decrypt_times': [],
            'query_results': []
        }
    
    def query_with_metrics(self,
                          database_server,
                          owner_id: int,
                          dataset_id: str,
                          model: Optional[List[float]] = None,
                          model_type: str = "linear") -> Tuple[Optional[List[float]], Dict]:
        """
        执行查询并测量所有阶段的时间
        
        返回:
            (results, metrics)
        """
        metrics = {
            'check_time': 0,
            'model_encrypt_time': 0,
            'query_time': 0,
            'decrypt_time': 0,
            'total_time': 0
        }
        
        start_total = time.time()
        
        # 1. 获取数据集
        C_m, sk_h_s = database_server.get_dataset(owner_id, dataset_id)
        if C_m is None:
            return None, metrics
        
        # 2. 检查权限
        start_check = time.time()
        C_M = self.check_access(C_m)
        metrics['check_time'] = time.time() - start_check
        
        if C_M is None:
            return None, metrics
        
        # 3. 准备模型
        if model is None:
            if C_m.get('c6_i') and len(C_m['c6_i']) > 0:
                try:
                    sample_data = self.he.decrypt(C_m['c6_i'][0])
                    dim = len(sample_data)
                except:
                    dim = 5
            else:
                dim = 5
            model = self.create_ai_model(model_type, dim)
        
        # 4. 加密模型
        start_encrypt = time.time()
        C_M = self.encrypt_ai_model(model, C_M)
        metrics['model_encrypt_time'] = time.time() - start_encrypt
        
        # 5. 执行查询
        start_query = time.time()
        ER = self.key_curator.system.query(C_M, C_m, sk_h_s)
        metrics['query_time'] = time.time() - start_query
        
        # 6. 解密结果
        start_decrypt = time.time()
        results = self.key_curator.system.decrypt(C_M['sk_h_u'], ER)
        metrics['decrypt_time'] = time.time() - start_decrypt
        
        metrics['total_time'] = time.time() - start_total
        metrics['num_results'] = len(results) if results else 0
        
        # 记录指标
        self.experiment_metrics['check_times'].append(metrics['check_time'])
        self.experiment_metrics['model_encrypt_times'].append(metrics['model_encrypt_time'])
        self.experiment_metrics['decrypt_times'].append(metrics['decrypt_time'])
        
        if results:
            self.experiment_metrics['query_results'].append({
                'num_results': len(results),
                'first_few': results[:3]
            })
        
        print(f"\n [DataQuerier {self.querier_id}] 查询阶段耗时:")
        print(f"   检查权限: {metrics['check_time']*1000:.2f} ms")
        print(f"   模型加密: {metrics['model_encrypt_time']*1000:.2f} ms")
        print(f"   查询执行: {metrics['query_time']*1000:.2f} ms")
        print(f"   结果解密: {metrics['decrypt_time']*1000:.2f} ms")
        print(f"   总计: {metrics['total_time']*1000:.2f} ms")
        
        return results, metrics
    
    def get_experiment_metrics(self) -> Dict:
        """获取实验指标统计"""
        metrics = {}
        
        if self.experiment_metrics['check_times']:
            metrics['avg_check_time'] = np.mean(self.experiment_metrics['check_times'])
        
        if self.experiment_metrics['model_encrypt_times']:
            metrics['avg_model_encrypt_time'] = np.mean(self.experiment_metrics['model_encrypt_times'])
        
        if self.experiment_metrics['decrypt_times']:
            metrics['avg_decrypt_time'] = np.mean(self.experiment_metrics['decrypt_times'])
        
        return metrics
    
    def reset_metrics(self):
        """重置实验指标"""
        self.experiment_metrics = {
            'check_times': [],
            'model_encrypt_times': [],
            'decrypt_times': [],
            'query_results': []
        }