"""
Measure pap_id size vs n to reveal DeCart O(n^2) vs DeCart* O(n) trend.

This benchmark varies n explicitly (instead of only varying N), so the
complexity difference is observable in per-user pap size.
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
from matplotlib.ticker import FormatStrFormatter

from accuracy_style import apply_accuracy_style, get_pic_accuracy_dir, single_output_path

apply_accuracy_style()

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from entities.key_curator import KeyCurator
from schemes.decart import DeCartParams
from schemes.decart_star import DeCartStarParams


class PapSizeVsNBenchmark:
    def __init__(self):
        self.results = {
            'decart': {'n': [], 'N': [], 'pap_avg': [], 'pap_std': []},
            'decart_star': {'n': [], 'N': [], 'pap_avg': [], 'pap_std': []},
        }
        self.results_dir = get_pic_accuracy_dir(project_root)

    def _resolve_N(self, n: int, n_mode: str, n_factor: int, n_fixed: int) -> int:
        if n_mode == 'equal':
            return n
        if n_mode == 'multiple':
            return max(n, n * n_factor)
        return max(n, n_fixed)

    def _measure_scheme(self, scheme: str, N: int, n: int, sample_users: int) -> Dict[str, float]:
        if scheme == 'decart':
            params = DeCartParams(N=N, n=n)
            curator = KeyCurator(scheme='decart', params=params)
        else:
            params = DeCartStarParams(N=N, n=n)
            curator = KeyCurator(scheme='decart_star', params=params)

        curator.setup()

        user_count = N if sample_users <= 0 else min(N, sample_users)
        pap_sizes = []
        for uid in range(user_count):
            sk, pk, pap = curator.generate_user_key(uid)
            curator.register(uid, pk, pap)
            pap_sizes.append(len(pickle.dumps(pap, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0)

        arr = np.array(pap_sizes, dtype=float)
        return {
            'pap_avg': float(np.mean(arr)) if arr.size else 0.0,
            'pap_std': float(np.std(arr)) if arr.size else 0.0,
        }

    def run(self, n_values: List[int], n_mode: str, n_factor: int, n_fixed: int, sample_users: int, show_plot: bool):
        print('\n' + '=' * 80)
        print('pap_id Size vs n Benchmark (DeCart vs DeCart*)')
        print('=' * 80)

        for n in n_values:
            N = self._resolve_N(n, n_mode, n_factor, n_fixed)
            print(f'\n--- n={n}, N={N} ---')

            d = self._measure_scheme('decart', N=N, n=n, sample_users=sample_users)
            s = self._measure_scheme('decart_star', N=N, n=n, sample_users=sample_users)

            self.results['decart']['n'].append(n)
            self.results['decart']['N'].append(N)
            self.results['decart']['pap_avg'].append(d['pap_avg'])
            self.results['decart']['pap_std'].append(d['pap_std'])

            self.results['decart_star']['n'].append(n)
            self.results['decart_star']['N'].append(N)
            self.results['decart_star']['pap_avg'].append(s['pap_avg'])
            self.results['decart_star']['pap_std'].append(s['pap_std'])

            ratio = d['pap_avg'] / s['pap_avg'] if s['pap_avg'] > 0 else 0.0
            print(f"DeCart={d['pap_avg']:.3f}KB, DeCart*={s['pap_avg']:.3f}KB, ratio={ratio:.2f}x")

        self._plot(show_plot=show_plot)
        self._print_summary()

    def _plot(self, show_plot: bool):
        n = np.array(self.results['decart']['n'], dtype=float)
        d_avg = np.array(self.results['decart']['pap_avg'], dtype=float)
        d_std = np.array(self.results['decart']['pap_std'], dtype=float)
        s_avg = np.array(self.results['decart_star']['pap_avg'], dtype=float)
        s_std = np.array(self.results['decart_star']['pap_std'], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(14, 8.2))

        ax1 = axes[0]
        ax1.errorbar(
            n,
            d_avg,
            yerr=d_std,
            fmt='o-',
            capsize=4,
            color='#3A5A98',
            markersize=6.5,
            markerfacecolor='white',
            markeredgewidth=1.4,
            linewidth=1.8,
            label='DeCart',
        )
        ax1.errorbar(
            n,
            s_avg,
            yerr=s_std,
            fmt='s-',
            capsize=4,
            color='#E07A5F',
            markersize=6.0,
            markerfacecolor='white',
            markeredgewidth=1.4,
            linewidth=1.8,
            label='DeCart*',
        )
        ax1.set_xlabel('n (users per block)')
        ax1.set_ylabel('pap_id Storage Size (KB)')
        ax1.set_title('Measured pap_id Storage Size vs n')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = axes[1]
        ratio = np.divide(d_avg, s_avg, out=np.zeros_like(d_avg), where=s_avg > 0)
        ax2.plot(
            n,
            ratio,
            'd-',
            color='#5FA8A3',
            linewidth=1.8,
            markersize=6.0,
            markerfacecolor='white',
            markeredgewidth=1.4,
        )
        for x, y in zip(n, ratio):
            ax2.text(x, y, f'{y:.4f}x', ha='center', va='bottom', fontsize=9)
        ax2.set_xlabel('n (users per block)')
        ax2.set_ylabel('Size Ratio (DeCart / DeCart*)')
        ax2.set_title('Size Ratio vs n')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        plt.tight_layout()
        out = single_output_path(self.results_dir, 'pap_size_vs_n', 'png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'\nSaved: {out}')

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _print_summary(self):
        print('\n' + '=' * 80)
        print('Summary (KB)')
        print('=' * 80)
        print(f"{'n':>6} {'N':>6} {'DeCart':>12} {'DeCart*':>12} {'Ratio':>10}")
        print('-' * 52)

        for i, n in enumerate(self.results['decart']['n']):
            N = self.results['decart']['N'][i]
            d = self.results['decart']['pap_avg'][i]
            s = self.results['decart_star']['pap_avg'][i]
            ratio = d / s if s > 0 else 0.0
            print(f'{n:6d} {N:6d} {d:12.3f} {s:12.3f} {ratio:10.2f}x')


def parse_n_values(text: str) -> List[int]:
    vals = []
    for part in text.split(','):
        v = int(part.strip())
        if v <= 0:
            raise ValueError('n values must be positive integers')
        vals.append(v)
    vals = sorted(set(vals))
    return vals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-values', type=str, default='8,16,24,32,48,64',
                        help='Comma-separated n list, e.g. 8,16,32,64')
    parser.add_argument('--n-mode', choices=['equal', 'multiple', 'fixed'], default='equal',
                        help='How to choose N for each n')
    parser.add_argument('--n-factor', type=int, default=4,
                        help='Used when n-mode=multiple, N=n*n_factor')
    parser.add_argument('--n-fixed', type=int, default=256,
                        help='Used when n-mode=fixed, N=max(n, n_fixed)')
    parser.add_argument('--sample-users', type=int, default=0,
                        help='Users sampled per (N,n), 0 means all users')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    n_values = parse_n_values(args.n_values)
    bench = PapSizeVsNBenchmark()
    bench.run(
        n_values=n_values,
        n_mode=args.n_mode,
        n_factor=args.n_factor,
        n_fixed=args.n_fixed,
        sample_users=args.sample_users,
        show_plot=(not args.no_show),
    )
