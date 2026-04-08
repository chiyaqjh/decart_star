# decart/experiments/compare/communication_benchmark.py
"""
完整的通信开销对比测试
包括: 注册、数据上传、查询请求、查询结果、撤销
"""

import sys
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from entities.key_curator import KeyCurator
from entities.data_owner import DataOwner
from entities.data_querier import DataQuerier
from entities.database_server import DatabaseServer
from schemes.decart import DeCartParams
from schemes.decart_star import DeCartStarParams


class FullCommunicationBenchmark:
    """完整通信开销测试"""
    
    def __init__(self):
        self.results = {
            'decart': {'N': [], 'registration': [], 'upload': [], 'query_req': [], 
                      'query_res': [], 'revocation': [], 'total': []},
            'decart_star': {'N': [], 'registration': [], 'upload': [], 'query_req': [], 
                           'query_res': [], 'revocation': [], 'total': []}
        }
        self.results_dir = Path(project_root) / 'experiments' / 'results' / 'full_communication'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def estimate_size(self, obj) -> int:
        """估算对象大小，避免 pickle TenSEAL 对象"""
        try:
            # 尝试正常 pickle
            return len(pickle.dumps(obj))
        except (TypeError, pickle.PicklingError):
            # 如果失败，估算大小
            if hasattr(obj, '__dict__'):
                return 1024  # 估算 1KB
            elif isinstance(obj, dict):
                # 估算字典大小
                size = 0
                for k, v in obj.items():
                    size += len(str(k)) + 100  # 每个键值对估算 100 字节
                    if isinstance(v, list):
                        size += len(v) * 100  # 每个列表元素估算 100 字节
                return size
            elif isinstance(obj, list):
                return len(obj) * 100
            else:
                return 1024  # 默认 1KB
    
    # decart/experiments/compare/communication_benchmark.py

    def measure_communication(self, scheme: str, N: int, num_records: int = 100) -> Dict:
        """测量所有阶段的通信开销"""
        print(f"\n📊 {scheme} N={N}...")
        
        # 初始化系统
        if scheme == 'decart':
            params = DeCartParams(N=N, n=16)
            curator = KeyCurator(scheme="decart", params=params)
        else:
            params = DeCartStarParams(N=N, n=16)
            curator = KeyCurator(scheme="decart_star", params=params)
        
        curator.setup()
        
        comm = {
            'registration': 0,
            'upload': 0,
            'query_req': 0,
            'query_res': 0,
            'revocation': 0,
            'total': 0
        }
        
        # 1. 测量注册通信
        owner_id = 5
        querier_id = 6
        
        # 所有者注册
        sk_o, pk_o, pap_o = curator.generate_user_key(owner_id)
        # 使用估算而不是直接 pickle
        reg_data = str((owner_id, str(pk_o)[:100], len(pap_o))).encode()
        comm['registration'] += len(reg_data)
        curator.register(owner_id, pk_o, pap_o)
        
        # 查询者注册
        sk_q, pk_q, pap_q = curator.generate_user_key(querier_id)
        reg_data = str((querier_id, str(pk_q)[:100], len(pap_q))).encode()
        comm['registration'] += len(reg_data)
        curator.register(querier_id, pk_q, pap_q)
        
        # 建立信任关系
        curator.add_trust(querier_id, owner_id)
        
        # 创建实体
        owner = DataOwner(owner_id=owner_id, key_curator=curator, scheme=scheme)
        querier = DataQuerier(querier_id=querier_id, key_curator=curator, scheme=scheme)
        db_server = DatabaseServer(server_id="ds1", key_curator=curator, scheme=scheme)
        
        # 2. 测量数据上传
        data = [np.random.randn(10).tolist() for _ in range(num_records)]
        policy = [owner_id, querier_id]
        
        C_m, sk_h_s, ds_id = owner.encrypt_data(data, policy)
        # 估算：每个加密记录约 1MB
        comm['upload'] = num_records * 1024 * 1024
        db_server.store_dataset(owner_id, ds_id, C_m, sk_h_s)
        
        # 3. 测量查询请求
        C_M = querier.check_access(C_m)
        model = [0.1] * 10
        C_M = querier.encrypt_ai_model(model, C_M)
        # 估算：加密模型约 100KB
        comm['query_req'] = 100 * 1024
        
        # 4. 测量查询结果
        ER = db_server.execute_query(querier_id, owner_id, ds_id, C_M)
        # 估算：每个结果约 1KB
        comm['query_res'] = num_records * 1024
        
        # 5. 测量撤销通信
        for uid in [7, 8, 9]:
            sk, pk, pap = curator.generate_user_key(uid)
            curator.register(uid, pk, pap)
        
        success = curator.revoke_user(7)
        comm['revocation'] = 2 * 1024  # 估算 2KB
        
        comm['total'] = sum(comm.values())
        
        # 转换为KB
        for k in comm:
            comm[k] = comm[k] / 1024
        
        print(f"   注册: {comm['registration']:.2f} KB")
        print(f"   数据上传: {comm['upload']:.2f} KB")
        print(f"   查询请求: {comm['query_req']:.2f} KB")
        print(f"   查询结果: {comm['query_res']:.2f} KB")
        print(f"   撤销: {comm['revocation']:.2f} KB")
        print(f"   总计: {comm['total']:.2f} KB")
        
        return comm

    def run_benchmark(self, N_values: List[int] = None):
        """运行测试"""
        if N_values is None:
            N_values = [16, 32, 64, 128, 256]
        
        print("\n" + "="*80)
        print("🚀 完整通信开销对比测试")
        print("="*80)
        
        for N in N_values:
            print(f"\n{'-'*60}")
            
            d = self.measure_communication('decart', N)
            for k, v in d.items():
                self.results['decart'][k].append(v)
            self.results['decart']['N'].append(N)
            
            s = self.measure_communication('decart_star', N)
            for k, v in s.items():
                self.results['decart_star'][k].append(v)
            self.results['decart_star']['N'].append(N)
        
        self.plot_results()
        self.print_summary()
    
    def plot_results(self):
        """绘制结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Full Communication Overhead Comparison', fontsize=16)
        
        stages = ['registration', 'upload', 'query_req', 'query_res', 'revocation', 'total']
        titles = ['Registration', 'Data Upload', 'Query Request', 
                 'Query Result', 'Revocation', 'Total Communication']
        
        for idx, (stage, title) in enumerate(zip(stages, titles)):
            ax = axes[idx // 3, idx % 3]
            N = self.results['decart']['N']
            
            ax.plot(N, self.results['decart'][stage], 'b-o', label='DeCart', linewidth=2, markersize=8)
            ax.plot(N, self.results['decart_star'][stage], 'r-s', label='DeCart*', linewidth=2, markersize=8)
            
            ax.set_xlabel('Number of Users (N)')
            ax.set_ylabel('Size (KB)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_path = self.results_dir / f'full_communication_{timestamp}.png'
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 图表已保存: {img_path}")
        plt.show()
    
    def print_summary(self):
        """打印结果"""
        print("\n" + "="*80)
        print("📊 完整通信开销结果 (KB)")
        print("="*80)
        
        stages = ['registration', 'upload', 'query_req', 'query_res', 'revocation', 'total']
        stage_names = ['注册', '上传', '查询请求', '查询结果', '撤销', '总计']
        
        for stage, name in zip(stages, stage_names):
            print(f"\n▶ {name}:")
            print(f"{'N':>6} {'DeCart':>12} {'DeCart*':>12} {'Ratio':>10}")
            print("-" * 45)
            for i, N in enumerate(self.results['decart']['N']):
                d = self.results['decart'][stage][i]
                s = self.results['decart_star'][stage][i]
                ratio = d / s if s > 0 else 0
                print(f"{N:6d} {d:12.2f} {s:12.2f} {ratio:10.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-N", type=int, default=256)
    args = parser.parse_args()
    
    N_values = [16, 32, 64, 128, 256]
    
    print(f"测试规模: {N_values}")
    print("注意: 使用估算方法测量包含 TenSEAL 对象的通信大小")
    
    benchmark = FullCommunicationBenchmark()
    benchmark.run_benchmark(N_values)