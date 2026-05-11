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

from accuracy_style import apply_accuracy_style, get_pic_accuracy_dir, single_output_path

apply_accuracy_style()

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
            'decart': {'N': [], 'pap_avg': [], 'pap_std': [], 'pap_min': [], 'pap_max': []},
            'decart_star': {'N': [], 'pap_avg': [], 'pap_std': [], 'pap_min': [], 'pap_max': []}
        }
        self.results_dir = get_pic_accuracy_dir(project_root)
    
    def measure_pap_size(self, scheme: str, N: int, n: int = 16, sample_users: int = 0) -> Dict:
        """测量用户 pap_id 大小统计（均值/方差）。"""
        print(f"\n📊 {scheme} N={N}...")
        
        # 初始化系统
        if scheme == 'decart':
            params = DeCartParams(N=N, n=n)
            curator = KeyCurator(scheme="decart", params=params)
        else:
            params = DeCartStarParams(N=N, n=n)
            curator = KeyCurator(scheme="decart_star", params=params)
        
        curator.setup()
        
        user_count = N if sample_users <= 0 else min(N, sample_users)
        pap_sizes_kb = []

        for uid in range(user_count):
            sk, pk, pap = curator.generate_user_key(uid)
            curator.register(uid, pk, pap)

            pap_bytes = pickle.dumps(pap, protocol=pickle.HIGHEST_PROTOCOL)
            pap_sizes_kb.append(len(pap_bytes) / 1024.0)

            if (uid + 1) % 50 == 0 or (uid + 1) == user_count:
                print(f"   已测量 {uid+1}/{user_count} 用户")

        pap_arr = np.array(pap_sizes_kb, dtype=float)
        stats = {
            'pap_avg': float(np.mean(pap_arr)) if pap_arr.size else 0.0,
            'pap_std': float(np.std(pap_arr)) if pap_arr.size else 0.0,
            'pap_min': float(np.min(pap_arr)) if pap_arr.size else 0.0,
            'pap_max': float(np.max(pap_arr)) if pap_arr.size else 0.0,
            'sample_users': user_count,
        }

        print(
            f"   pap 平均: {stats['pap_avg']:.2f} KB, "
            f"标准差: {stats['pap_std']:.4f}, "
            f"范围: [{stats['pap_min']:.2f}, {stats['pap_max']:.2f}] KB"
        )
        return stats

    def run_benchmark(self, N_values: List[int] = None, sample_users: int = 0, show_plot: bool = True):
        """运行测试"""
        if N_values is None:
            N_values = [16, 32, 64, 128, 256, 512]
        
        print("\n" + "="*80)
        print("🚀 验证用户存储 O(n²) vs O(n)")
        print("="*80)
        
        for N in N_values:
            # DeCart
            stats = self.measure_pap_size('decart', N, sample_users=sample_users)
            self.results['decart']['N'].append(N)
            self.results['decart']['pap_avg'].append(stats['pap_avg'])
            self.results['decart']['pap_std'].append(stats['pap_std'])
            self.results['decart']['pap_min'].append(stats['pap_min'])
            self.results['decart']['pap_max'].append(stats['pap_max'])
            
            # DeCart*
            stats = self.measure_pap_size('decart_star', N, sample_users=sample_users)
            self.results['decart_star']['N'].append(N)
            self.results['decart_star']['pap_avg'].append(stats['pap_avg'])
            self.results['decart_star']['pap_std'].append(stats['pap_std'])
            self.results['decart_star']['pap_min'].append(stats['pap_min'])
            self.results['decart_star']['pap_max'].append(stats['pap_max'])
        
        self.plot_results(show_plot=show_plot)
        self.print_summary()
    
    def plot_results(self, show_plot: bool = True):
        """绘制结果（单用户均值 + 随N总量图）"""
        plt.figure(figsize=(12, 8))
        
        # 实测均值与标准差误差棒
        N = np.array(self.results['decart']['N'])
        d_avg = np.array(self.results['decart']['pap_avg'])
        d_std = np.array(self.results['decart']['pap_std'])
        s_avg = np.array(self.results['decart_star']['pap_avg'])
        s_std = np.array(self.results['decart_star']['pap_std'])

        plt.errorbar(N, d_avg, yerr=d_std, fmt='b-o', capsize=4,
                     label='DeCart (mean ± std)', linewidth=2, markersize=8)
        plt.errorbar(N, s_avg, yerr=s_std, fmt='r-s', capsize=4,
                     label='DeCart* (mean ± std)', linewidth=2, markersize=8)

        for x, y in zip(N, d_avg):
            plt.text(x, y, f'{y:.2f}', color='b', fontsize=9, ha='center', va='bottom')
        for x, y in zip(N, s_avg):
            plt.text(x, y, f'{y:.2f}', color='r', fontsize=9, ha='center', va='top')

        plt.xlabel('Number of Users (N)', fontsize=14)
        plt.ylabel('pap_id Size (KB)', fontsize=14)
        plt.title('User Storage: pap_id Size by N (Measured)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 保存
        img_path = single_output_path(self.results_dir, 'user_storage', 'png')
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 图表已保存: {img_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()

        # 新增：与 N 相关的总 pap 存储图（按均值估算总量）
        plt.figure(figsize=(12, 8))
        total_d = N * d_avg
        total_s = N * s_avg
        plt.plot(N, total_d, 'b-o', linewidth=2, markersize=8, label='DeCart total pap (estimated)')
        plt.plot(N, total_s, 'r-s', linewidth=2, markersize=8, label='DeCart* total pap (estimated)')

        for x, y in zip(N, total_d):
            plt.text(x, y, f'{y:.1f}', color='b', fontsize=9, ha='center', va='bottom')
        for x, y in zip(N, total_s):
            plt.text(x, y, f'{y:.1f}', color='r', fontsize=9, ha='center', va='bottom')

        plt.xlabel('Number of Users (N)', fontsize=14)
        plt.ylabel('Total pap Storage (KB)', fontsize=14)
        plt.title('Estimated Total pap Storage by N (N × mean pap size)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        total_img_path = single_output_path(self.results_dir, 'user_storage_total_by_N', 'png')
        plt.savefig(total_img_path, dpi=150, bbox_inches='tight')
        print(f"📊 图表已保存: {total_img_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def print_summary(self):
        """打印结果"""
        print("\n" + "="*80)
        print("📊 用户 pap_id 存储统计 (KB)")
        print("="*80)
        print(f"{'N':>6} {'DeCart(avg±std)':>20} {'DeCart*(avg±std)':>20} {'Ratio(avg)':>12}")
        print("-" * 64)
        
        for i, N in enumerate(self.results['decart']['N']):
            d = self.results['decart']['pap_avg'][i]
            d_std = self.results['decart']['pap_std'][i]
            s = self.results['decart_star']['pap_avg'][i]
            s_std = self.results['decart_star']['pap_std'][i]
            ratio = d / s if s > 0 else 0
            print(f"{N:6d} {d:9.2f}±{d_std:7.3f} {s:9.2f}±{s_std:7.3f} {ratio:12.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-N", type=int, default=512)
    parser.add_argument("--sample-users", type=int, default=0,
                        help="每个N测量的用户数量；0表示测量全部N个用户")
    parser.add_argument("--no-show", action="store_true",
                        help="仅保存图片，不弹出窗口")
    args = parser.parse_args()
    
    N_values = [16, 32, 64, 128, 256, 512]
    N_values = [n for n in N_values if n <= args.max_N]
    
    benchmark = UserStorageBenchmark()
    benchmark.run_benchmark(N_values, sample_users=args.sample_users, show_plot=(not args.no_show))