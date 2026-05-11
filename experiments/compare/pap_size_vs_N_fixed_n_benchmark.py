"""
Benchmark pap_id size under fixed n and varying N.

Goal:
- Keep n fixed (default n=16)
- Vary N and measure pap_id size statistics for DeCart and DeCart*
- Save figure to experiments/results/pic_accuracy
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from accuracy_style import apply_accuracy_style, get_pic_accuracy_dir, single_output_path

apply_accuracy_style()

# Add project root to import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from entities.key_curator import KeyCurator
from schemes.decart import DeCartParams
from schemes.decart_star import DeCartStarParams


class PapSizeVsNFixedNBenchmark:
    def __init__(self):
        self.results = {
            'decart': {'N': [], 'pap_avg': [], 'pap_std': []},
            'decart_star': {'N': [], 'pap_avg': [], 'pap_std': []},
        }
        self.results_dir = get_pic_accuracy_dir(project_root)

    def _measure_scheme(self, scheme: str, N: int, n: int, sample_users: int) -> Dict[str, float]:
        if scheme == 'decart':
            params = DeCartParams(N=N, n=n)
            curator = KeyCurator(scheme='decart', params=params)
        else:
            params = DeCartStarParams(N=N, n=n)
            curator = KeyCurator(scheme='decart_star', params=params)

        curator.setup()

        user_count = N if sample_users <= 0 else min(N, sample_users)
        pap_sizes_kb = []

        for uid in range(user_count):
            sk, pk, pap = curator.generate_user_key(uid)
            curator.register(uid, pk, pap)
            pap_sizes_kb.append(len(pickle.dumps(pap, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0)

        arr = np.array(pap_sizes_kb, dtype=float)
        return {
            'pap_avg': float(np.mean(arr)) if arr.size else 0.0,
            'pap_std': float(np.std(arr)) if arr.size else 0.0,
        }

    def run(self, N_values: List[int], n: int, sample_users: int, show_plot: bool):
        print('\n' + '=' * 80)
        print(f'pap_id Size vs N (fixed n={n})')
        print('=' * 80)

        for N in N_values:
            if N < n:
                print(f'Skip N={N}: must satisfy N >= n={n}')
                continue

            print(f'\n--- N={N}, n={n} ---')
            d = self._measure_scheme('decart', N=N, n=n, sample_users=sample_users)
            s = self._measure_scheme('decart_star', N=N, n=n, sample_users=sample_users)

            self.results['decart']['N'].append(N)
            self.results['decart']['pap_avg'].append(d['pap_avg'])
            self.results['decart']['pap_std'].append(d['pap_std'])

            self.results['decart_star']['N'].append(N)
            self.results['decart_star']['pap_avg'].append(s['pap_avg'])
            self.results['decart_star']['pap_std'].append(s['pap_std'])

            ratio = d['pap_avg'] / s['pap_avg'] if s['pap_avg'] > 0 else 0.0
            print(f"DeCart={d['pap_avg']:.3f}KB, DeCart*={s['pap_avg']:.3f}KB, ratio={ratio:.2f}x")

        self._plot(n=n, show_plot=show_plot)
        self._print_summary(n=n)

    def _plot(self, n: int, show_plot: bool):
        N = np.array(self.results['decart']['N'], dtype=float)
        d_avg = np.array(self.results['decart']['pap_avg'], dtype=float)
        d_std = np.array(self.results['decart']['pap_std'], dtype=float)
        s_avg = np.array(self.results['decart_star']['pap_avg'], dtype=float)
        s_std = np.array(self.results['decart_star']['pap_std'], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        ax1.errorbar(N, d_avg, yerr=d_std, fmt='o-', capsize=4, label='DeCart (mean ± std)')
        ax1.errorbar(N, s_avg, yerr=s_std, fmt='s-', capsize=4, label='DeCart* (mean ± std)')
        ax1.set_xlabel('N (max users)')
        ax1.set_ylabel('pap_id size (KB)')
        ax1.set_title(f'pap_id Size vs N (fixed n={n})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = axes[1]
        ratio = np.divide(d_avg, s_avg, out=np.zeros_like(d_avg), where=s_avg > 0)
        ax2.plot(N, ratio, 'd-', color='purple')
        for x, y in zip(N, ratio):
            ax2.text(x, y, f'{y:.2f}x', ha='center', va='bottom', fontsize=9)
        ax2.set_xlabel('N (max users)')
        ax2.set_ylabel('DeCart / DeCart*')
        ax2.set_title('Size Ratio vs N')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out = single_output_path(self.results_dir, f'pap_size_vs_N_fixed_n{n}', 'png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'\nSaved: {out}')

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _print_summary(self, n: int):
        print('\n' + '=' * 80)
        print(f'Summary (fixed n={n}, KB)')
        print('=' * 80)
        print(f"{'N':>6} {'DeCart(avg±std)':>20} {'DeCart*(avg±std)':>20} {'Ratio':>10}")
        print('-' * 62)

        for i, N in enumerate(self.results['decart']['N']):
            d = self.results['decart']['pap_avg'][i]
            d_std = self.results['decart']['pap_std'][i]
            s = self.results['decart_star']['pap_avg'][i]
            s_std = self.results['decart_star']['pap_std'][i]
            ratio = d / s if s > 0 else 0.0
            print(f'{int(N):6d} {d:9.3f}±{d_std:7.3f} {s:9.3f}±{s_std:7.3f} {ratio:10.2f}x')


def parse_int_list(text: str) -> List[int]:
    vals = []
    for part in text.split(','):
        v = int(part.strip())
        if v <= 0:
            raise ValueError('N values must be positive integers')
        vals.append(v)
    return sorted(set(vals))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N-values', type=str, default='16,32,64,128,256',
                        help='Comma-separated N list, e.g. 16,32,64,128,256')
    parser.add_argument('--n', type=int, default=16, help='Fixed n value')
    parser.add_argument('--sample-users', type=int, default=0,
                        help='Users sampled per N, 0 means all users')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    N_values = parse_int_list(args.N_values)
    bench = PapSizeVsNFixedNBenchmark()
    bench.run(N_values=N_values, n=args.n, sample_users=args.sample_users, show_plot=(not args.no_show))
