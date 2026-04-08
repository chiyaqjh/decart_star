# decart/experiments/our_decart/server.py
"""
DeCart 方案数据库服务器实验接口
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

from entities.database_server import DatabaseServer as BaseDatabaseServer


class DatabaseServer(BaseDatabaseServer):
    """
    DeCart 方案数据库服务器（实验版）
    添加实验测量功能
    """
    
    def __init__(self, server_id: str, key_curator, scheme: str = "decart"):
        super().__init__(server_id, key_curator, scheme)
        
        # 实验测量数据
        self.experiment_metrics = {
            'query_times': [],
            'result_sizes': [],
            'model_types': [],
            'dataset_access_times': []
        }
    
    def execute_query_with_metrics(self,
                                querier_id: int,
                                owner_id: int,
                                dataset_id: str,
                                C_M: Dict) -> Tuple[Optional[Dict], Dict]:
        """
        执行查询并测量指标
        
        返回:
            (ER, metrics)
        """
        # 记录开始时间
        start = time.time()
        
        # 执行查询
        ER = self.execute_query(querier_id, owner_id, dataset_id, C_M)
        
        # 计算查询时间
        query_time = time.time() - start
        
        metrics = {
            'query_time': query_time,
            'query_time_ms': query_time * 1000,
            'success': ER is not None
        }
        
        if ER is not None:
            # 测量结果大小 - 使用估算方法
            try:
                import pickle
                result_size = len(pickle.dumps(ER))
            except (TypeError, pickle.PicklingError):
                # 估算大小
                if 'encrypted_results' in ER:
                    result_size = len(ER['encrypted_results']) * 1024 * 1024  # 1MB per result
                else:
                    result_size = 1024 * 1024  # 默认 1MB
            
            metrics['result_size'] = result_size
            metrics['num_results'] = ER.get('num_results', 0)
            
            self.experiment_metrics['result_sizes'].append(result_size)
            
            # 记录模型类型
            model_type = C_M.get('model_type', 'unknown')
            self.experiment_metrics['model_types'].append(model_type)
            
            print(f"\n [DatabaseServer] 查询指标:")
            print(f"   查询时间: {query_time*1000:.2f} ms")
            print(f"   结果大小: {result_size/1024:.2f} KB")
            print(f"   结果数量: {metrics['num_results']}")
        
        self.experiment_metrics['query_times'].append(query_time)
        
        return ER, metrics
    
    def get_experiment_metrics(self) -> Dict:
        """获取实验指标统计"""
        metrics = {}
        
        if self.experiment_metrics['query_times']:
            times = self.experiment_metrics['query_times']
            metrics['avg_query_time'] = np.mean(times)
            metrics['std_query_time'] = np.std(times)
            metrics['min_query_time'] = np.min(times)
            metrics['max_query_time'] = np.max(times)
        
        if self.experiment_metrics['result_sizes']:
            sizes = self.experiment_metrics['result_sizes']
            metrics['avg_result_size'] = np.mean(sizes)
            metrics['total_results_size'] = sum(sizes)
        
        # 按模型类型统计
        if self.experiment_metrics['model_types']:
            from collections import Counter
            model_counts = Counter(self.experiment_metrics['model_types'])
            metrics['model_type_counts'] = dict(model_counts)
        
        return metrics
    
    def reset_metrics(self):
        """重置实验指标"""
        self.experiment_metrics = {
            'query_times': [],
            'result_sizes': [],
            'model_types': [],
            'dataset_access_times': []
        }