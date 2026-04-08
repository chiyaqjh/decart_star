# decart/experiments/compare/user_storage_benchmark.py
"""
验证每个用户存储的 pap_id 大小随 N 的变化
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


class UserStorageBenchmark:
    """测量每个用户的 pap_id 存储大小"""
    
    def __init__(self):
        self.results = {
            'decart': {'N': [], 'pap_size': []},
            'decart_star': {'N': [], 'pap_size': []}
        }
        self.results_dir = Path(project_root) / 'experiments' / 'results' / 'user_storage'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def measure_pap_size(self, scheme: str, N: int, n: int = 16) -> float:
        """测量单个用户的 pap_id 大小"""
        print(f"\n📊 {scheme} N={N}...")
        
        # 初始化系统
        if scheme == 'decart':
            params = DeCartParams(N=N, n=n)
            curator = KeyCurator(scheme="decart", params=params)
        else:
            params = DeCartStarParams(N=N, n=n)
            curator = KeyCurator(scheme="decart_star", params=params)
        
        curator.setup()
        
        # 注册一个用户
        test_uid = 5
        sk, pk, pap = curator.generate_user_key(test_uid)
        curator.register(test_uid, pk, pap)
        
        # 测量 pap_id 大小
        pap_bytes = pickle.dumps(pap)
        pap_size_kb = len(pap_bytes) / 1024
        
        print(f"   pap_id 大小: {pap_size_kb:.2f} KB")
        return pap_size_kb
    
    def run_benchmark(self, N_values: List[int] = None):
        """运行测试"""
        if N_values is None:
            N_values = [16, 32, 64, 128, 256, 512]
        
        print("\n" + "="*80)
        print("🚀 验证用户存储 O(n²) vs O(n)")
        print("="*80)
        
        for N in N_values:
            # DeCart
            size = self.measure_pap_size('decart', N)
            self.results['decart']['N'].append(N)
            self.results['decart']['pap_size'].append(size)
            
            # DeCart*
            size = self.measure_pap_size('decart_star', N)
            self.results['decart_star']['N'].append(N)
            self.results['decart_star']['pap_size'].append(size)
        
        self.plot_results()
        self.print_summary()
    
    def plot_results(self):
        """绘制结果"""
        plt.figure(figsize=(12, 8))
        
        # 实际数据
        plt.plot(self.results['decart']['N'], self.results['decart']['pap_size'], 
                'b-o', label='DeCart (actual)', linewidth=2, markersize=8)
        plt.plot(self.results['decart_star']['N'], self.results['decart_star']['pap_size'], 
                'r-s', label='DeCart* (actual)', linewidth=2, markersize=8)
        
        # 理论曲线拟合
        N = np.array(self.results['decart']['N'])
        
        # 对 DeCart 拟合二次曲线
        coeffs = np.polyfit(N, self.results['decart']['pap_size'], 2)
        poly = np.poly1d(coeffs)
        N_smooth = np.linspace(min(N), max(N), 100)
        plt.plot(N_smooth, poly(N_smooth), 'b--', 
                label=f'DeCart quadratic fit: {coeffs[0]:.2e}N²', alpha=0.7)
        
        # 对 DeCart* 拟合线性曲线
        coeffs_star = np.polyfit(N, self.results['decart_star']['pap_size'], 1)
        linear = np.poly1d(coeffs_star)
        plt.plot(N_smooth, linear(N_smooth), 'r--', 
                label=f'DeCart* linear fit: {coeffs_star[0]:.2f}N', alpha=0.7)
        
        plt.xlabel('Number of Users (N)', fontsize=14)
        plt.ylabel('pap_id Size (KB)', fontsize=14)
        plt.title('User Storage: DeCart (O(n²)) vs DeCart* (O(n))', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_path = self.results_dir / f'user_storage_{timestamp}.png'
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 图表已保存: {img_path}")
        plt.show()
    
    def print_summary(self):
        """打印结果"""
        print("\n" + "="*80)
        print("📊 用户存储大小 (KB)")
        print("="*80)
        print(f"{'N':>6} {'DeCart':>12} {'DeCart*':>12} {'Ratio':>10}")
        print("-" * 45)
        
        for i, N in enumerate(self.results['decart']['N']):
            d = self.results['decart']['pap_size'][i]
            s = self.results['decart_star']['pap_size'][i]
            ratio = d / s if s > 0 else 0
            print(f"{N:6d} {d:12.2f} {s:12.2f} {ratio:10.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-N", type=int, default=512)
    args = parser.parse_args()
    
    N_values = [16, 32, 64, 128, 256, 512]
    N_values = [n for n in N_values if n <= args.max_N]
    
    benchmark = UserStorageBenchmark()
    benchmark.run_benchmark(N_values)