# decart/experiments/compare/CRS_total_storage.py
"""
验证 Key Curator 总存储和 CRS 大小
DeCart: O(n²) vs DeCart*: O(n)
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
from schemes.decart import DeCartParams
from schemes.decart_star import DeCartStarParams


class TotalStorageBenchmark:
    """测量 Key Curator 总存储和 CRS 大小"""
    
    def __init__(self):
        self.results = {
            'decart': {'N': [], 'total_storage': [], 'crs_size': [], 'avg_pap': []},
            'decart_star': {'N': [], 'total_storage': [], 'crs_size': [], 'avg_pap': []}
        }
        self.results_dir = Path(project_root) / 'experiments' / 'results' / 'total_storage'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def estimate_crs_size(self, scheme: str, N: int, n: int = 16) -> float:
        """估算 CRS 大小（KB）- 避免 pickle 模块对象"""
        if scheme == 'decart':
            # DeCart: h_i (n个) + H_ij (n*n个)
            n_elements = n + (n * n)
            # 每个元素估算 256 字节
            size_bytes = n_elements * 256
        else:
            # DeCart*: h_i (2n个)
            n_elements = 2 * n
            size_bytes = n_elements * 256
        
        return size_bytes / 1024  # 转换为 KB
    
    def measure_storage(self, scheme: str, N: int, n: int = 16) -> Dict:
        """测量 Key Curator 总存储和 CRS 大小"""
        print(f"\n📊 {scheme} N={N}...")
        
        # 初始化系统
        if scheme == 'decart':
            params = DeCartParams(N=N, n=n)
            curator = KeyCurator(scheme="decart", params=params)
        else:
            params = DeCartStarParams(N=N, n=n)
            curator = KeyCurator(scheme="decart_star", params=params)
        
        curator.setup()
        
        # 估算 CRS 大小（不直接 pickle）
        crs_size_kb = self.estimate_crs_size(scheme, N, n)
        
        # 注册所有 N 个用户
        total_pap_size = 0
        for uid in range(N):
            sk, pk, pap = curator.generate_user_key(uid)
            curator.register(uid, pk, pap)
            
            # pap 是列表，可以 pickle
            pap_bytes = pickle.dumps(pap)
            total_pap_size += len(pap_bytes)
            
            if (uid + 1) % 50 == 0:
                print(f"   已注册 {uid+1}/{N} 用户")
        
        # 总存储 = CRS + 所有用户的 pap
        total_storage_kb = crs_size_kb + (total_pap_size / 1024)
        avg_pap_kb = (total_pap_size / N) / 1024 if N > 0 else 0
        
        print(f"   CRS (估算): {crs_size_kb:.2f} KB")
        print(f"   所有用户 pap: {total_pap_size/1024:.2f} KB")
        print(f"   总存储: {total_storage_kb:.2f} KB")
        print(f"   平均 pap: {avg_pap_kb:.2f} KB")
        
        return {
            'crs_size': crs_size_kb,
            'total_storage': total_storage_kb,
            'avg_pap': avg_pap_kb
        }
    
    def run_benchmark(self, N_values: List[int] = None):
        """运行测试"""
        if N_values is None:
            N_values = [16, 32, 64, 128, 256, 512]
        
        print("\n" + "="*80)
        print("🚀 验证 Key Curator 总存储和 CRS 大小")
        print("="*80)
        
        for N in N_values:
            print(f"\n{'-'*60}")
            
            # DeCart
            d = self.measure_storage('decart', N)
            self.results['decart']['N'].append(N)
            self.results['decart']['crs_size'].append(d['crs_size'])
            self.results['decart']['total_storage'].append(d['total_storage'])
            self.results['decart']['avg_pap'].append(d['avg_pap'])
            
            # DeCart*
            s = self.measure_storage('decart_star', N)
            self.results['decart_star']['N'].append(N)
            self.results['decart_star']['crs_size'].append(s['crs_size'])
            self.results['decart_star']['total_storage'].append(s['total_storage'])
            self.results['decart_star']['avg_pap'].append(s['avg_pap'])
        
        self.plot_results()
        self.print_summary()
    
    def plot_results(self):
        """绘制结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        N = np.array(self.results['decart']['N'])
        
        # 图1: CRS 大小
        ax1 = axes[0]
        ax1.plot(N, self.results['decart']['crs_size'], 'b-o', label='DeCart', linewidth=2, markersize=8)
        ax1.plot(N, self.results['decart_star']['crs_size'], 'r-s', label='DeCart*', linewidth=2, markersize=8)
        
        # 拟合曲线
        if len(N) > 2:
            coeffs_d = np.polyfit(N, self.results['decart']['crs_size'], 2)
            coeffs_s = np.polyfit(N, self.results['decart_star']['crs_size'], 1)
            N_smooth = np.linspace(min(N), max(N), 100)
            ax1.plot(N_smooth, np.poly1d(coeffs_d)(N_smooth), 'b--', alpha=0.5, 
                    label=f'{coeffs_d[0]:.2e}N²')
            ax1.plot(N_smooth, np.poly1d(coeffs_s)(N_smooth), 'r--', alpha=0.5,
                    label=f'{coeffs_s[0]:.2f}N')
        
        ax1.set_xlabel('Number of Users (N)')
        ax1.set_ylabel('CRS Size (KB)')
        ax1.set_title('CRS Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 总存储
        ax2 = axes[1]
        ax2.plot(N, self.results['decart']['total_storage'], 'b-o', label='DeCart', linewidth=2, markersize=8)
        ax2.plot(N, self.results['decart_star']['total_storage'], 'r-s', label='DeCart*', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Number of Users (N)')
        ax2.set_ylabel('Total Storage (KB)')
        ax2.set_title('Key Curator Total Storage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图3: 平均 pap 大小
        ax3 = axes[2]
        ax3.plot(N, self.results['decart']['avg_pap'], 'b-o', label='DeCart', linewidth=2, markersize=8)
        ax3.plot(N, self.results['decart_star']['avg_pap'], 'r-s', label='DeCart*', linewidth=2, markersize=8)
        
        ax3.set_xlabel('Number of Users (N)')
        ax3.set_ylabel('Average pap Size (KB)')
        ax3.set_title('Average pap Size per User')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_path = self.results_dir / f'total_storage_{timestamp}.png'
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 图表已保存: {img_path}")
        plt.show()
    
    def print_summary(self):
        """打印结果"""
        print("\n" + "="*80)
        print("📊 Key Curator 存储结果 (KB)")
        print("="*80)
        
        print("\n▶ CRS 大小 (估算):")
        print(f"{'N':>6} {'DeCart':>12} {'DeCart*':>12} {'Ratio':>10}")
        print("-" * 45)
        for i, N in enumerate(self.results['decart']['N']):
            d = self.results['decart']['crs_size'][i]
            s = self.results['decart_star']['crs_size'][i]
            print(f"{N:6d} {d:12.2f} {s:12.2f} {d/s:10.2f}x")
        
        print("\n▶ 总存储 (CRS + 所有用户 pap):")
        print(f"{'N':>6} {'DeCart':>12} {'DeCart*':>12} {'Ratio':>10}")
        print("-" * 45)
        for i, N in enumerate(self.results['decart']['N']):
            d = self.results['decart']['total_storage'][i]
            s = self.results['decart_star']['total_storage'][i]
            print(f"{N:6d} {d:12.2f} {s:12.2f} {d/s:10.2f}x")
        
        print("\n▶ 平均 pap 大小 (每个用户):")
        print(f"{'N':>6} {'DeCart':>12} {'DeCart*':>12} {'Ratio':>10}")
        print("-" * 45)
        for i, N in enumerate(self.results['decart']['N']):
            d = self.results['decart']['avg_pap'][i]
            s = self.results['decart_star']['avg_pap'][i]
            print(f"{N:6d} {d:12.2f} {s:12.2f} {d/s:10.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-N", type=int, default=256)
    args = parser.parse_args()
    
    N_values = [16, 32, 64, 128, 256]
    if args.max_N > 256:
        N_values.extend([512])
    
    print(f"测试规模: {N_values}")
    print("注意: 这会注册所有用户，可能需要较长时间")
    
    benchmark = TotalStorageBenchmark()
    benchmark.run_benchmark(N_values)