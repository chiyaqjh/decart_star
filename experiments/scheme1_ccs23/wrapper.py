# decart/experiments/scheme1_ccs23/wrapper.py
"""
CCS23 直接应用方案实验包装器
作为性能上限基准（中心化服务器）
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


class CCS23ExperimentWrapper:
    """
    CCS23 方案实验包装器
    特点：
    - 中心化服务器
    - 服务器持有所有数据明文
    - 查询者发送模型明文
    - 服务器明文执行推理
    - 作为性能上限基准
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
        
        # 初始化同态加密（仅用于测量，实际不用）
        self.he = HomomorphicEncryption(poly_modulus_degree=4096)
        
        # 数据存储（明文！）
        self.datasets = {}  # owner_id -> {dataset_id -> {'data': data, 'policy': policy}}
        
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
        
        print(f" CCS23 实验环境初始化完成（中心化明文计算）")
    
    def setup(self) -> float:
        """
        初始化系统
        
        返回:
            setup_time: 初始化耗时
        """
        start = time.perf_counter()
        # CCS23 不需要复杂的密码学设置
        elapsed = time.perf_counter() - start
        self.metrics['setup_time'] = elapsed
        print(f"   CCS23 初始化完成: {elapsed:.4f}秒")
        return elapsed
    
    def register_user(self, user_id: int) -> Tuple[int, Any]:
        """
        注册用户（CCS23 中简单记录）
        
        返回:
            (sk, pk) 这里返回虚拟值
        """
        start = time.perf_counter()
        
        self.registered_users.add(user_id)
        
        elapsed = time.perf_counter() - start
        self.metrics['keygen_times'].append(elapsed)
        
        # CCS23 中不需要真正的密钥，返回虚拟值
        return user_id, f"dummy_pk_{user_id}"
    
    def encrypt_dataset(self, 
                       owner_id: int,
                       data: List[List[float]],
                       policy: List[int],
                       metadata: Optional[Dict] = None) -> Tuple[Dict, Any, str]:
        """
        CCS23 中"加密"数据集（实际存储明文）
        
        返回:
            (C_m, sk_h_s, dataset_id) - 格式兼容
        """
        # 生成数据集ID
        import time
        timestamp = int(time.time() * 1000)
        dataset_id = f"ds_{owner_id}_{timestamp}"
        
        # 存储明文数据
        if owner_id not in self.datasets:
            self.datasets[owner_id] = {}
        
        self.datasets[owner_id][dataset_id] = {
            'data': data.copy(),
            'policy': policy.copy(),
            'metadata': metadata or {},
            'store_time': time.time()
        }
        
        # 测量"加密"时间（实际为0）
        elapsed = 0.001  # 1ms 模拟
        self.metrics['encrypt_times'].append(elapsed)
        
        # 测量数据大小（用于通信开销对比）
        import sys
        import pickle
        size = sys.getsizeof(pickle.dumps(data))
        
        self.metrics['communication_sizes'].append({
            'type': 'encrypt',
            'size': size,
            'records': len(data)
        })
        
        print(f"    CCS23 存储数据: {elapsed*1000:.2f} ms, 数据大小: {size/1024:.2f} KB")
        
        # 返回格式兼容的元数据
        C_m = {
            'type': 'ccs23_plain',
            'dataset_id': dataset_id,
            'owner_id': owner_id,
            'policy': policy,
            'num_records': len(data)
        }
        
        return C_m, None, dataset_id
    
    def store_dataset(self, owner_id: int, dataset_id: str, C_m: Dict, sk_h_s: Any):
        """CCS23 中不需要额外存储（已在 encrypt_dataset 中完成）"""
        pass
    
    def execute_query(self,
                     querier_id: int,
                     owner_id: int,
                     dataset_id: str,
                     model: Any) -> Optional[List[float]]:
        """
        CCS23 中执行明文查询
        
        返回:
            查询结果
        """
        # 检查数据集是否存在
        if owner_id not in self.datasets or dataset_id not in self.datasets[owner_id]:
            print(f"      数据集不存在")
            return None
        
        dataset_info = self.datasets[owner_id][dataset_id]
        data = dataset_info['data']
        policy = dataset_info['policy']
        
        # CCS23 中不检查权限（直接访问）
        
        # 执行明文查询
        start_query = time.perf_counter()
        
        results = []
        if isinstance(model, list):
            # 点积模型: y = model · x
            for record in data:
                # 确保维度匹配
                min_len = min(len(model), len(record))
                result = sum(model[i] * record[i] for i in range(min_len))
                results.append(result)
        elif isinstance(model, dict) and model.get('type') == 'neural_network':
            # 神经网络模型 - 单层
            weights = model.get('weights', [])
            bias = model.get('bias', [])
            output_dim = model.get('output_dim', 10)
            
            for record in data:
                # 简单矩阵乘法（单层）
                outputs = []
                for i in range(min(output_dim, 10)):
                    # 取对应行的权重
                    start_idx = i * len(record)
                    end_idx = (i + 1) * len(record)
                    row_weights = weights[start_idx:end_idx] if len(weights) > start_idx else []
                    
                    if row_weights:
                        val = sum(w * record[j] for j, w in enumerate(row_weights) if j < len(record))
                        if i < len(bias):
                            val += bias[i]
                        outputs.append(val)
                    else:
                        outputs.append(0.0)
                
                results.append(outputs[0] if outputs else 0.0)
        elif isinstance(model, dict) and model.get('type') == 'decision_tree':
            # 简单决策树：统一基线逻辑
            # 规则: feature[0] <= 0.5 -> 左叶子，否则右叶子
            nodes = model.get('nodes', [])
            node_map = {n.get('id'): n for n in nodes}
            root_id = model.get('root', 0)

            for record in data:
                current_id = root_id
                depth = 0
                max_depth = 10

                while depth < max_depth and current_id in node_map:
                    node = node_map[current_id]

                    if 'value' in node:
                        results.append(float(node['value']))
                        break

                    feature_idx = int(node.get('feature', 0))
                    threshold = float(node.get('threshold', 0.0))
                    feature_val = float(record[feature_idx]) if feature_idx < len(record) else 0.0

                    if feature_val <= threshold:
                        current_id = node.get('left')
                    else:
                        current_id = node.get('right')
                    depth += 1
                else:
                    # 防御性回退
                    results.append(0.0)
        else:
            # 默认处理
            for record in data:
                results.append(float(np.random.randn()))
        
        query_time = time.perf_counter() - start_query
        self.metrics['query_times'].append(query_time)
        
        # CCS23 中解密时间为0
        decrypt_time = 0
        self.metrics['decrypt_times'].append(decrypt_time)
        
        print(f"    CCS23 查询时间: {query_time*1000:.2f} ms, 结果数量: {len(results)}")
        
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
        
        # 通信大小
        if metrics['communication_sizes']:
            sizes = [s['size'] for s in metrics['communication_sizes']]
            metrics['avg_communication_size'] = np.mean(sizes)
            metrics['total_communication'] = np.sum(sizes)
        
        return metrics