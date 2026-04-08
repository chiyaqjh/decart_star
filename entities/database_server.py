# decart/entities/database_server.py
"""
Database Server 实体 - 论文第I.A节
作为服务提供商，存储数据记录，执行加密AI查询
支持DeCart和DeCart*双方案 + 撤销后数据集更新 + 预训练模型查询统计
完全非模拟，基于真实同态加密
"""

import sys
import os
import copy
import time
import importlib
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.homomorphic import HomomorphicEncryption
from entities.key_curator import KeyCurator
from entities.data_owner import DataOwner
from entities.data_querier import DataQuerier
from schemes.decart_star import DeCartStarParams


class DatabaseServer:
    """
    数据库服务器 (Database Server)
    
    论文职责:
    1. 存储数据所有者的加密数据集
    2. 接收数据查询者的加密AI模型
    3. 执行加密AI查询 - Query(C_M, C_m) → ER
    4. 返回加密查询结果
    5. 支持撤销后数据集的更新
    6. 拒绝被撤销用户的查询
    7. 统计不同模型类型的查询
    
    支持双方案:
    - DeCart  : 原始方案，O(n²)复杂度
    - DeCart* : 优化方案，O(n)复杂度，20倍性能提升
    
    安全要求:
    - 无法解密任何数据（只有密文）
    - 无法获取AI模型明文
    - 半可信假设：好奇但遵守协议
    - 被撤销用户无法执行查询
    
    完全非模拟:
    - 真实同态加密 (TenSEAL CKKS)
    - 真实双线性配对 (bn256)
    """
    
    def __init__(self,
                 server_id: str = "ds1",
                 key_curator: Optional[KeyCurator] = None,
                 scheme: str = "decart_star"):
        """
        初始化数据库服务器
        
        参数:
            server_id: 服务器唯一标识
            key_curator: 密钥管理者实例（可选）
            scheme: "decart" 或 "decart_star"
        """
        self.server_id = server_id
        self.key_curator = key_curator
        self.scheme = scheme.lower()
        
        # 存储结构
        self.datasets = {}           # owner_id -> {dataset_id -> {'C_m': , 'sk_h_s': , 'metadata': , 'valid': bool}}
        self.query_logs = []         # 查询日志
        self.access_logs = []        # 访问日志
        self.revoked_queries_blocked = 0  # 被阻止的撤销用户查询计数
        
        # ===== 新增：模型类型统计 =====
        self.model_type_stats = defaultdict(int)  # 各类型模型查询次数
        self.model_query_times = defaultdict(list)  # 各类型模型查询耗时
        
        # 数据集版本跟踪
        self.dataset_versions = {}   # owner_id -> {dataset_id -> version}
        
        # 性能统计
        self.stats = {
            'total_datasets': 0,
            'total_queries': 0,
            'total_computation_time': 0,
            'blocked_revoked_queries': 0,
            'dataset_updates': 0,
            'start_time': time.time(),
            # 新增统计
            'dot_queries': 0,
            'decision_tree_queries': 0,
            'neural_network_queries': 0,
            'unknown_model_queries': 0
        }
        
        print(f"\n  Database Server 实体初始化")
        print(f"   服务器ID: {server_id}")
        print(f"   方案: {self.scheme_name() if key_curator else scheme}")
        print(f"   存储容量: 无限")
        print(f"   支持撤销检查")
        print(f"   支持模型类型统计")
    
    def scheme_name(self) -> str:
        """获取当前使用的方案名称"""
        if self.key_curator:
            return self.key_curator.scheme_name
        return "DeCart*" if self.scheme == "decart_star" else "DeCart"
    
    # ========== 撤销检查 ==========
    
    def _check_querier_revoked(self, querier_id: int) -> bool:
        """
        检查查询者是否已被撤销
        
        参数:
            querier_id: 查询者ID
        
        返回:
            True: 已被撤销, False: 正常
        """
        if self.key_curator and self.key_curator.is_revoked(querier_id):
            self.revoked_queries_blocked += 1
            self.stats['blocked_revoked_queries'] += 1
            print(f"   ⚠️ 查询者 {querier_id} 已被撤销，拒绝查询")
            return True
        return False
    
    def _check_dataset_valid(self, owner_id: int, dataset_id: str) -> bool:
        """
        检查数据集是否有效（未被标记为无效）
        
        参数:
            owner_id: 所有者ID
            dataset_id: 数据集ID
        
        返回:
            True: 有效, False: 无效
        """
        if owner_id not in self.datasets:
            return False
        
        if dataset_id not in self.datasets[owner_id]:
            return False
        
        dataset_info = self.datasets[owner_id][dataset_id]
        return dataset_info.get('valid', True)
    
    # ========== 数据存储接口 ==========
    
    def store_dataset(self,
                     owner_id: int,
                     dataset_id: str,
                     C_m: Dict,
                     sk_h_s: Any,
                     metadata: Optional[Dict] = None) -> bool:
        """
        存储加密数据集
        
        参数:
            owner_id: 数据所有者ID
            dataset_id: 数据集ID
            C_m: 加密数据集密文
            sk_h_s: 同态加密服务端密钥份额
            metadata: 元数据
        
        返回:
            存储是否成功
        """
        print(f"\n[Database Server {self.server_id}] 存储数据集")
        print(f"   所有者: {owner_id}")
        print(f"   数据集: {dataset_id}")
        print(f"   数据记录数: {len(C_m.get('c6_i', []))}")
        
        # 初始化所有者的存储空间
        if owner_id not in self.datasets:
            self.datasets[owner_id] = {}
            print(f"   创建所有者 {owner_id} 的存储空间")
        
        # 检查是否已存在
        if dataset_id in self.datasets[owner_id]:
            print(f"   ⚠️  数据集已存在，将覆盖")
        
        # 获取当前版本号
        if owner_id not in self.dataset_versions:
            self.dataset_versions[owner_id] = {}
        version = self.dataset_versions[owner_id].get(dataset_id, 0) + 1
        
        # 存储数据集
        self.datasets[owner_id][dataset_id] = {
            'C_m': C_m,
            'sk_h_s': sk_h_s,
            'metadata': metadata or {},
            'store_time': time.time(),
            'owner_id': owner_id,
            'dataset_id': dataset_id,
            'record_count': len(C_m.get('c6_i', [])),
            'access_count': 0,
            'valid': True,
            'version': version,
            'original_policy': C_m.get('P', []).copy()
        }
        
        self.dataset_versions[owner_id][dataset_id] = version
        self.stats['total_datasets'] += 1
        
        print(f"     数据集存储成功")
        print(f"      版本: {version}")
        print(f"      所有者 {owner_id} 现有数据集: {list(self.datasets[owner_id].keys())}")
        print(f"      当前总数据集: {self.stats['total_datasets']}")
        
        return True
    
    def update_dataset(self,
                      owner_id: int,
                      dataset_id: str,
                      C_m_new: Dict,
                      sk_h_s_new: Optional[Any] = None,
                      metadata: Optional[Dict] = None) -> bool:
        """
        更新已存储的数据集（撤销后更新）
        
        参数:
            owner_id: 数据所有者ID
            dataset_id: 数据集ID
            C_m_new: 新的加密数据集密文
            sk_h_s_new: 新的同态加密密钥份额（如果改变）
            metadata: 更新的元数据
        
        返回:
            更新是否成功
        """
        print(f"\n[Database Server {self.server_id}] 更新数据集")
        print(f"   所有者: {owner_id}")
        print(f"   数据集: {dataset_id}")
        
        if owner_id not in self.datasets:
            print(f"     所有者 {owner_id} 不存在")
            return False
        
        if dataset_id not in self.datasets[owner_id]:
            print(f"     数据集 {dataset_id} 不存在")
            return False
        
        # 获取原数据集信息
        old_info = self.datasets[owner_id][dataset_id]
        old_policy = old_info['C_m'].get('P', [])
        new_policy = C_m_new.get('P', [])
        
        print(f"   原策略: {old_policy}")
        print(f"   新策略: {new_policy}")
        
        # 获取新版本号
        version = self.dataset_versions[owner_id].get(dataset_id, 0) + 1
        
        # 更新数据集
        self.datasets[owner_id][dataset_id] = {
            'C_m': C_m_new,
            'sk_h_s': sk_h_s_new if sk_h_s_new is not None else old_info['sk_h_s'],
            'metadata': {**old_info['metadata'], **(metadata or {})},
            'store_time': time.time(),
            'owner_id': owner_id,
            'dataset_id': dataset_id,
            'record_count': len(C_m_new.get('c6_i', [])),
            'access_count': old_info['access_count'],
            'valid': True,
            'version': version,
            'previous_version': old_info.get('version', 0),
            'original_policy': old_policy,
            'updated_after_revoke': True,
            'removed_users': [u for u in old_policy if u not in new_policy]
        }
        
        self.dataset_versions[owner_id][dataset_id] = version
        self.stats['dataset_updates'] += 1
        
        print(f"     数据集更新成功")
        print(f"      新版本: {version}")
        print(f"      移除用户: {[u for u in old_policy if u not in new_policy]}")
        
        return True
    
    def mark_dataset_invalid(self, owner_id: int, dataset_id: str, reason: str = "") -> bool:
        """
        将数据集标记为无效（当策略为空时）
        
        参数:
            owner_id: 所有者ID
            dataset_id: 数据集ID
            reason: 无效原因
        
        返回:
            标记是否成功
        """
        if owner_id not in self.datasets:
            return False
        
        if dataset_id not in self.datasets[owner_id]:
            return False
        
        self.datasets[owner_id][dataset_id]['valid'] = False
        self.datasets[owner_id][dataset_id]['invalid_reason'] = reason
        self.datasets[owner_id][dataset_id]['invalid_time'] = time.time()
        
        print(f"\n[Database Server] 数据集 {dataset_id} 已标记为无效")
        print(f"   原因: {reason}")
        
        return True
    
    def batch_store_datasets(self,
                           datasets: List[Tuple[int, str, Dict, Any, Optional[Dict]]]) -> int:
        """
        批量存储多个数据集
        
        参数:
            datasets: [(owner_id, dataset_id, C_m, sk_h_s, metadata), ...]
        
        返回:
            成功存储的数量
        """
        success_count = 0
        for owner_id, dataset_id, C_m, sk_h_s, metadata in datasets:
            if self.store_dataset(owner_id, dataset_id, C_m, sk_h_s, metadata):
                success_count += 1
        
        print(f"\n[Database Server] 批量存储完成: {success_count}/{len(datasets)}")
        return success_count
    
    def get_dataset(self, owner_id: int, dataset_id: str) -> Tuple[Optional[Dict], Optional[Any]]:
        """
        获取加密数据集（检查有效性）
        
        参数:
            owner_id: 数据所有者ID
            dataset_id: 数据集ID
        
        返回:
            (C_m, sk_h_s) 或 (None, None)
        """
        if owner_id not in self.datasets:
            print(f"     所有者 {owner_id} 没有存储任何数据集")
            return None, None
        
        if dataset_id not in self.datasets[owner_id]:
            print(f"     数据集 {dataset_id} 不存在")
            print(f"   可用数据集: {list(self.datasets[owner_id].keys())}")
            return None, None
        
        dataset = self.datasets[owner_id][dataset_id]
        
        # 检查数据集是否有效
        if not dataset.get('valid', True):
            print(f"   ⚠️ 数据集 {dataset_id} 已失效")
            print(f"      原因: {dataset.get('invalid_reason', '未知')}")
            return None, None
        
        dataset['access_count'] += 1
        
        return dataset['C_m'], dataset['sk_h_s']
    
    def list_datasets(self, owner_id: Optional[int] = None, include_invalid: bool = False) -> List[Dict]:
        """
        列出数据集
        
        参数:
            owner_id: 指定所有者，None时列出所有
            include_invalid: 是否包含已失效的数据集
        """
        result = []
        
        if owner_id is not None:
            # 列出特定所有者的数据集
            if owner_id in self.datasets:
                for ds_id, info in self.datasets[owner_id].items():
                    if not include_invalid and not info.get('valid', True):
                        continue
                    result.append({
                        'owner_id': owner_id,
                        'dataset_id': ds_id,
                        'record_count': info['record_count'],
                        'store_time': info['store_time'],
                        'access_count': info['access_count'],
                        'metadata': info['metadata'],
                        'valid': info.get('valid', True),
                        'version': info.get('version', 0),
                        'policy': info['C_m'].get('P', [])
                    })
        else:
            # 列出所有数据集
            for oid in self.datasets:
                for ds_id, info in self.datasets[oid].items():
                    if not include_invalid and not info.get('valid', True):
                        continue
                    result.append({
                        'owner_id': oid,
                        'dataset_id': ds_id,
                        'record_count': info['record_count'],
                        'store_time': info['store_time'],
                        'access_count': info['access_count'],
                        'metadata': info['metadata'],
                        'valid': info.get('valid', True),
                        'version': info.get('version', 0),
                        'policy': info['C_m'].get('P', [])
                    })
        
        return sorted(result, key=lambda x: x['store_time'], reverse=True)
    
    def delete_dataset(self, owner_id: int, dataset_id: str) -> bool:
        """
        删除数据集
        """
        if owner_id not in self.datasets:
            print(f"     所有者 {owner_id} 不存在")
            return False
        
        if dataset_id not in self.datasets[owner_id]:
            print(f"     数据集 {dataset_id} 不存在")
            return False
        
        del self.datasets[owner_id][dataset_id]
        if not self.datasets[owner_id]:
            del self.datasets[owner_id]
        
        self.stats['total_datasets'] -= 1
        print(f"[Database Server] 数据集 {dataset_id} 已删除")
        print(f"   剩余数据集数: {self.stats['total_datasets']}")
        
        return True
    
    # ========== 论文算法：Query（委托给对应方案）==========
    
    def execute_query(self,
                     querier_id: int,
                     owner_id: int,
                     dataset_id: str,
                     C_M: Dict) -> Optional[Dict]:
        """
        执行加密AI查询 - 论文Query(C_M, C_m)算法
        增强版：统计不同模型类型的查询
        
        参数:
            querier_id: 查询者ID
            owner_id: 数据所有者ID
            dataset_id: 数据集ID
            C_M: 查询者的加密AI模型和密钥
        
        返回:
            ER: 加密查询结果
        """
        print(f"\n[Database Server {self.server_id}] 执行加密查询")
        print(f"   查询者: {querier_id}")
        print(f"   所有者: {owner_id}")
        print(f"   数据集: {dataset_id}")
        
        # 1. 检查查询者是否已被撤销
        if self._check_querier_revoked(querier_id):
            print(f"     查询者已被撤销，拒绝执行")
            
            self.query_logs.append({
                'timestamp': time.time(),
                'querier_id': querier_id,
                'owner_id': owner_id,
                'dataset_id': dataset_id,
                'status': 'blocked_revoked',
                'error': 'Querier revoked'
            })
            
            return None
        
        # 2. 获取数据集
        C_m, sk_h_s = self.get_dataset(owner_id, dataset_id)
        if C_m is None:
            print(f"     数据集不存在或已失效")
            return None
        
        # 3. 验证访问权限（由系统Check算法保证）
        if not C_M.get('access_granted', False):
            print(f"     未授权访问")
            return None
        
        # 4. 判断模型类型
        encrypted_model = C_M.get('encrypted_model', {})
        if isinstance(encrypted_model, dict):
            model_type = encrypted_model.get('type', 'unknown')
        else:
            model_type = 'dot_product'
        
        print(f"   模型类型: {model_type}")
        
        # 5. 执行加密查询
        try:
            start_time = time.time()
            
            # 调用对应方案的Query算法
            ER = self.key_curator.system.query(C_M, C_m, sk_h_s)
            
            query_time = time.time() - start_time
            
            # 6. 更新统计
            self.model_type_stats[model_type] += 1
            self.model_query_times[model_type].append(query_time)
            
            if model_type == 'decision_tree':
                self.stats['decision_tree_queries'] += 1
            elif model_type == 'neural_network':
                self.stats['neural_network_queries'] += 1
            elif model_type == 'dot_product':
                self.stats['dot_queries'] += 1
            else:
                self.stats['unknown_model_queries'] += 1
            
            # 7. 记录日志
            self.query_logs.append({
                'timestamp': time.time(),
                'querier_id': querier_id,
                'owner_id': owner_id,
                'dataset_id': dataset_id,
                'query_time': query_time,
                'result_count': len(ER.get('encrypted_results', [])),
                'model_type': model_type,
                'status': 'success'
            })
            
            self.stats['total_queries'] += 1
            self.stats['total_computation_time'] += query_time
            
            print(f"     查询执行成功")
            print(f"      执行时间: {query_time*1000:.2f} ms")
            print(f"      结果数量: {len(ER.get('encrypted_results', []))}")
            print(f"      模型类型: {model_type}")
            
            return ER
            
        except Exception as e:
            print(f"     查询执行失败: {e}")
            
            self.query_logs.append({
                'timestamp': time.time(),
                'querier_id': querier_id,
                'owner_id': owner_id,
                'dataset_id': dataset_id,
                'error': str(e),
                'model_type': model_type,
                'status': 'failed'
            })
            
            return None
    
    # ========== 批量查询 ==========
    
    def batch_execute_queries(self,
                            queries: List[Tuple[int, int, str, Dict]]) -> List[Optional[Dict]]:
        """
        批量执行多个查询
        
        参数:
            queries: [(querier_id, owner_id, dataset_id, C_M), ...]
        
        返回:
            查询结果列表
        """
        print(f"\n[Database Server] 批量执行 {len(queries)} 个查询")
        
        results = []
        blocked_count = 0
        
        for querier_id, owner_id, dataset_id, C_M in queries:
            # 检查每个查询者
            if self._check_querier_revoked(querier_id):
                results.append(None)
                blocked_count += 1
                continue
                
            ER = self.execute_query(querier_id, owner_id, dataset_id, C_M)
            results.append(ER)
        
        success_count = sum(1 for r in results if r is not None)
        print(f"     批量查询完成: {success_count}/{len(queries)} 成功")
        print(f"      {blocked_count} 个被撤销用户被阻止")
        
        return results
    
    # ========== 模型类型统计 ==========
    
    def get_model_type_stats(self) -> Dict:
        """
        获取各模型类型的查询统计
        
        返回:
            {
                'dot_product': 次数,
                'decision_tree': 次数,
                'neural_network': 次数,
                'unknown': 次数,
                'avg_times': {类型: 平均耗时}
            }
        """
        avg_times = {}
        for model_type, times in self.model_query_times.items():
            if times:
                avg_times[model_type] = sum(times) / len(times)
            else:
                avg_times[model_type] = 0
        
        return {
            'dot_product': self.stats['dot_queries'],
            'decision_tree': self.stats['decision_tree_queries'],
            'neural_network': self.stats['neural_network_queries'],
            'unknown': self.stats['unknown_model_queries'],
            'total': self.stats['total_queries'],
            'avg_times': avg_times,
            'raw_stats': dict(self.model_type_stats)
        }
    
    def print_model_stats(self):
        """打印模型类型统计信息"""
        print("\n" + "="*60)
        print("📊 模型查询统计")
        print("="*60)
        
        stats = self.get_model_type_stats()
        
        print(f"   点积模型: {stats['dot_product']} 次")
        print(f"   决策树模型: {stats['decision_tree']} 次")
        print(f"   神经网络模型: {stats['neural_network']} 次")
        print(f"   未知类型: {stats['unknown']} 次")
        print(f"   总计: {stats['total']} 次")
        
        if stats['avg_times']:
            print(f"\n   平均查询时间:")
            for model_type, avg_time in stats['avg_times'].items():
                if avg_time > 0:
                    print(f"     {model_type}: {avg_time*1000:.2f} ms")
    
    # ========== 查询日志与统计 ==========
    
    def get_query_logs(self, limit: int = 100, include_blocked: bool = True) -> List[Dict]:
        """
        获取查询日志
        
        参数:
            limit: 返回数量限制
            include_blocked: 是否包含被阻止的查询
        """
        logs = self.query_logs
        if not include_blocked:
            logs = [log for log in logs if log.get('status') != 'blocked_revoked']
        
        return sorted(
            logs[-limit:],
            key=lambda x: x['timestamp'],
            reverse=True
        )
    
    def get_access_logs(self, limit: int = 100) -> List[Dict]:
        """
        获取访问日志
        """
        logs = []
        for owner_id in self.datasets:
            for ds_id, info in self.datasets[owner_id].items():
                if info.get('valid', True):
                    logs.append({
                        'timestamp': info['store_time'],
                        'owner_id': owner_id,
                        'dataset_id': ds_id,
                        'action': 'store',
                        'record_count': info['record_count'],
                        'version': info.get('version', 0)
                    })
        
        logs.extend(self.query_logs)
        
        return sorted(logs[-limit:], key=lambda x: x['timestamp'], reverse=True)
    
    def get_server_stats(self) -> Dict:
        """
        获取服务器统计信息
        """
        total_records = 0
        valid_datasets = 0
        invalid_datasets = 0
        
        for owner_id in self.datasets:
            for ds_id, info in self.datasets[owner_id].items():
                total_records += info['record_count']
                if info.get('valid', True):
                    valid_datasets += 1
                else:
                    invalid_datasets += 1
        
        # 获取模型统计
        model_stats = self.get_model_type_stats()
        
        return {
            'server_id': self.server_id,
            'scheme': self.scheme_name(),
            'total_datasets': self.stats['total_datasets'],
            'valid_datasets': valid_datasets,
            'invalid_datasets': invalid_datasets,
            'total_queries': self.stats['total_queries'],
            'blocked_revoked_queries': self.stats['blocked_revoked_queries'],
            'dataset_updates': self.stats['dataset_updates'],
            'total_records': total_records,
            'unique_owners': len(self.datasets),
            'avg_query_time': (self.stats['total_computation_time'] / self.stats['total_queries']) 
                             if self.stats['total_queries'] > 0 else 0,
            'model_stats': model_stats,
            'uptime': time.time() - self.stats['start_time']
        }
    
    def clear_logs(self):
        """清除所有日志"""
        self.query_logs = []
        self.access_logs = []
        self.revoked_queries_blocked = 0
        self.model_type_stats.clear()
        self.model_query_times.clear()
        print(f"[Database Server] 日志已清除")
    
    def clear_all_data(self):
        """清除所有数据（用于测试）"""
        self.datasets = {}
        self.query_logs = []
        self.access_logs = []
        self.dataset_versions = {}
        self.model_type_stats.clear()
        self.model_query_times.clear()
        self.stats['total_datasets'] = 0
        self.stats['total_queries'] = 0
        self.stats['total_computation_time'] = 0
        self.stats['blocked_revoked_queries'] = 0
        self.stats['dataset_updates'] = 0
        self.stats['dot_queries'] = 0
        self.stats['decision_tree_queries'] = 0
        self.stats['neural_network_queries'] = 0
        self.stats['unknown_model_queries'] = 0
        self.revoked_queries_blocked = 0
        print(f"[Database Server] 所有数据已清除")


# ========== 测试代码 ==========

# decart/entities/database_server.py (修复测试函数中的方法调用)

def test_database_server_model_stats():
    """测试数据库服务器的模型统计功能"""
    
    print("\n" + "="*80)
    print("🧪 测试 Database Server 模型统计功能")
    print("="*80)
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.data_querier import DataQuerier
    from schemes.decart_star import DeCartStarParams
    import glob
    
    # 1. 初始化系统
    print("\n1. 初始化系统...")
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. 创建用户
    print("\n2. 创建用户...")
    owner_id = 5
    querier_id = 6
    
    sk_o, pk_o, pap_o = curator.generate_user_key(owner_id)
    curator.register(owner_id, pk_o, pap_o)
    
    sk_q, pk_q, pap_q = curator.generate_user_key(querier_id)
    curator.register(querier_id, pk_q, pap_q)
    
    # 3. 建立信任关系
    print("\n3. 建立信任关系...")
    curator.add_trust(querier_id, owner_id)
    
    # 4. 数据所有者加密数据
    print("\n4. 数据所有者加密数据...")
    owner = DataOwner(owner_id=owner_id, key_curator=curator, scheme="decart_star")
    data = [np.random.randn(5).tolist() for _ in range(3)]
    policy = [owner_id, querier_id]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy)
    
    # 5. 创建数据库服务器
    print("\n5. 创建数据库服务器...")
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(owner_id, ds_id, C_m, sk_h_s)
    
    # 6. 创建查询者
    print("\n6. 创建查询者...")
    querier = DataQuerier(querier_id=querier_id, key_curator=curator, scheme="decart_star")
    
    # 7. 执行不同类型的查询
    print("\n7. 执行不同类型的查询...")
    
    # 点积查询
    print(f"\n   [点积查询]")
    C_M_base = querier.check_access(C_m)
    model = [0.1, 0.2, 0.3, 0.4, 0.5]
    C_M = querier.encrypt_ai_model(model, C_M_base)
    db_server.execute_query(querier_id, owner_id, ds_id, C_M)
    
    # 神经网络查询（模拟）
    print(f"\n   [神经网络查询]")
    C_M_base = querier.check_access(C_m)
    C_M_base['encrypted_model'] = {'type': 'neural_network', 'layers': []}
    C_M_base['access_granted'] = True
    db_server.execute_query(querier_id, owner_id, ds_id, C_M_base)
    
    # 决策树查询（模拟）
    print(f"\n   [决策树查询]")
    C_M_base = querier.check_access(C_m)
    C_M_base['encrypted_model'] = {'type': 'decision_tree', 'nodes': []}
    C_M_base['access_granted'] = True
    db_server.execute_query(querier_id, owner_id, ds_id, C_M_base)
    
    # 8. 查看统计
    print("\n8. 查看模型统计...")
    db_server.print_model_stats()
    
    # 修复：使用正确的方法名 get_model_type_stats()
    stats = db_server.get_model_type_stats()
    print(f"\n   统计结果: {stats}")
    
    assert stats['dot_product'] == 1, f"点积查询应为1，实际为{stats['dot_product']}"
    assert stats['neural_network'] == 1, f"神经网络查询应为1，实际为{stats['neural_network']}"
    assert stats['decision_tree'] == 1, f"决策树查询应为1，实际为{stats['decision_tree']}"
    
    print(f"\n  Database Server 模型统计测试通过")
    
    return db_server

def test_database_server_revoke_handling():
    """测试Database Server的撤销处理功能"""
    
    print("\n" + "="*80)
    print(" 测试 Database Server 撤销处理")
    print("="*80)
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.data_querier import DataQuerier
    from schemes.decart_star import DeCartStarParams
    
    # 1. 初始化系统
    print("\n1. 初始化系统...")
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. 创建用户
    print("\n2. 创建用户...")
    users = [5, 6, 7, 8]
    user_keys = {}
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)
        user_keys[uid] = {'sk': sk, 'pk': pk, 'pap': pap}
    
    # 3. 建立信任关系
    print("\n3. 建立信任关系...")
    curator.add_trust(6, 5)
    curator.add_trust(7, 5)
    curator.add_trust(8, 5)
    
    # 4. 创建数据所有者
    print("\n4. 创建数据所有者...")
    owner = DataOwner(owner_id=5, key_curator=curator, scheme="decart_star")
    
    # 5. 加密数据集
    print("\n5. 加密数据集...")
    data = [[1.0, 2.0, 3.0, 4.0, 5.0]] * 3
    policy = [5, 6, 7, 8]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy, store_original=True)
    
    # 6. 创建数据库服务器
    print("\n6. 创建数据库服务器...")
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(5, ds_id, C_m, sk_h_s, {'name': 'test'})
    
    # 7. 创建查询者
    print("\n7. 创建查询者...")
    querier6 = DataQuerier(querier_id=6, key_curator=curator, scheme="decart_star")
    querier7 = DataQuerier(querier_id=7, key_curator=curator, scheme="decart_star")
    
    # 8. 正常查询
    print("\n8. 正常用户执行查询...")
    C_M6 = querier6.check_access(C_m)
    model6 = querier6.create_ai_model(dimension=5)
    C_M6 = querier6.encrypt_ai_model(model6, C_M6)
    result6 = db_server.execute_query(6, 5, ds_id, C_M6)
    assert result6 is not None, "正常用户查询应成功"
    print(f"     用户6查询成功")
    
    # 9. 撤销用户6
    print("\n9. 撤销用户6...")
    curator.revoke_user(6)
    
    # 10. 尝试让已撤销用户6查询（应失败）
    print("\n10. 已撤销用户尝试查询（应失败）...")
    result6_after = db_server.execute_query(6, 5, ds_id, C_M6)
    assert result6_after is None, "被撤销用户查询应失败"
    print(f"     被撤销用户查询被拒绝")
    
    # 11. 检查统计信息
    print("\n11. 检查统计信息...")
    stats = db_server.get_server_stats()
    print(f"   总查询数: {stats['total_queries']}")
    print(f"   被阻止的撤销查询: {stats['blocked_revoked_queries']}")
    assert stats['blocked_revoked_queries'] >= 1, "应有被阻止的撤销查询"
    
    print(f"\n  Database Server 撤销处理测试通过")
    
    return db_server


def test_database_server_batch_revoke():
    """测试批量查询中的撤销处理"""
    
    print("\n" + "="*80)
    print(" 测试 Database Server 批量撤销处理")
    print("="*80)
    
    from entities.key_curator import KeyCurator
    from entities.data_owner import DataOwner
    from entities.data_querier import DataQuerier
    from schemes.decart_star import DeCartStarParams
    
    # 1. 初始化系统
    curator = KeyCurator(scheme="decart_star", params=DeCartStarParams(N=64, n=16))
    curator.setup()
    
    # 2. 创建用户
    users = [5, 6, 7, 8]
    user_keys = {}
    for uid in users:
        sk, pk, pap = curator.generate_user_key(uid)
        curator.register(uid, pk, pap)
        user_keys[uid] = {'sk': sk, 'pk': pk, 'pap': pap}
    
    # 3. 建立信任关系
    for uid in [6, 7, 8]:
        curator.add_trust(uid, 5)
    
    # 4. 数据所有者加密数据
    owner = DataOwner(owner_id=5, key_curator=curator, scheme="decart_star")
    data = [[1.0, 2.0, 3.0]]
    policy = [5, 6, 7, 8]
    C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy)
    
    # 5. 数据库服务器
    db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme="decart_star")
    db_server.store_dataset(5, ds_id, C_m, sk_h_s)
    
    # 6. 创建所有查询者的C_M
    queries = []
    for uid in [6, 7, 8]:
        querier = DataQuerier(querier_id=uid, key_curator=curator, scheme="decart_star")
        C_M = querier.check_access(C_m)
        model = querier.create_ai_model(dimension=3)
        C_M = querier.encrypt_ai_model(model, C_M)
        queries.append((uid, 5, ds_id, C_M))
    
    # 7. 撤销用户7
    print("\n7. 撤销用户7...")
    curator.revoke_user(7)
    
    # 8. 批量执行查询
    print("\n8. 批量执行查询...")
    results = db_server.batch_execute_queries(queries)
    
    # 9. 验证结果
    print("\n9. 验证结果...")
    success_count = sum(1 for r in results if r is not None)
    print(f"   成功查询数: {success_count}")
    print(f"   总查询数: {len(queries)}")
    assert success_count == 2, "应有2个成功查询（用户6和8）"
    assert results[1] is None, "用户7的查询应被拒绝"
    
    print(f"\n  批量撤销处理测试通过")


if __name__ == "__main__":
    print("="*80)
    print("🔬 Database Server 实体测试套件")
    print("="*80)
    
    # 运行原有测试
    test_database_server_revoke_handling()
    test_database_server_batch_revoke()
    
    # 运行新增的模型统计测试
    test_database_server_model_stats()
    
    print("\n" + "="*80)
    print("  所有 Database Server 测试通过")
    print("   完全非模拟，真实密码学")
    print("   支持 DeCart / DeCart* 双方案")
    print("   支持撤销检查和数据集更新")
    print("   支持模型类型统计")
    print("="*80)   
  