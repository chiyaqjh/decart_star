"""
Generate five-scheme comparison figures with configuration-matched result files.

This avoids mixing files from different sizes (e.g. CCS23=32 while others=128).
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from accuracy_style import apply_accuracy_style, get_pic_accuracy_dir, single_output_path

apply_accuracy_style()


ROOT = 'E:/decart/experiments/results'
PIC_DIR = str(get_pic_accuracy_dir('E:/decart'))

SCHEME_FOLDERS = {
    'CCS23': 'scheme1_ccs23',
    'Server': 'scheme2_server',
    'Offline': 'scheme3_offline',
    'DeCart': 'our_decart',
    'DeCart*': 'our_decart_star',
}

MODEL_KEYS = ['dot', 'decision_tree', 'neural_network']
MODEL_TITLES = ['Dot Product', 'Decision Tree', 'Neural Network']
PHASES = [
    ('encrypt_times', 'Encrypt Time (ms)'),
    ('query_times', 'Query Time (ms)'),
    ('decrypt_times', 'Decrypt Time (ms)'),
]


def _load_matched_result(folder, num_records, record_dim, policy_size, num_runs):
    files = sorted(glob.glob(os.path.join(ROOT, folder, '*.json')), reverse=True)
    for path in files:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = data.get('config', {})
        if (
            cfg.get('num_records') == num_records
            and cfg.get('record_dim') == record_dim
            and cfg.get('policy_size') == policy_size
            and cfg.get('num_runs') == num_runs
        ):
            return path, data
    return None, None


def _mean_ms(block, key):
    vals = block.get(key, [])
    return float(np.mean(vals)) * 1000.0 if vals else 0.0


def _save_path(prefix):
    os.makedirs(PIC_DIR, exist_ok=True)
    return str(single_output_path(Path(PIC_DIR), prefix, 'png'))


def generate_3x3(loaded):
    schemes = list(SCHEME_FOLDERS.keys())

    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5), sharex=False)
    fig.suptitle('Five-Scheme Comparison', fontsize=20, fontweight='bold')

    phase_styles = {
        'encrypt_times': {'edgecolor': '#3A5A98', 'hatch': '/////'},
        'query_times': {'edgecolor': '#5FA8A3', 'hatch': '.....'},
        'decrypt_times': {'edgecolor': '#E07A5F', 'hatch': 'xxxxx'},
    }

    for r, (mk, mt) in enumerate(zip(MODEL_KEYS, MODEL_TITLES)):
        for c, (metric_key, phase_label) in enumerate(PHASES):
            ax = axes[r, c]
            vals_ms = [_mean_ms(loaded[s]['models'][mk], metric_key) for s in schemes]
            style = phase_styles[metric_key]

            bars = ax.bar(
                schemes,
                vals_ms,
                color='none',
                edgecolor=style['edgecolor'],
                linewidth=1.4,
                hatch=style['hatch'],
            )
            ax.set_title(f"{mt} - {phase_label.replace(' (ms)', '')}", fontsize=13)
            ax.grid(axis='y', alpha=0.25)
            ax.tick_params(axis='x', labelrotation=15)
            ax.tick_params(axis='y', labelsize=11)

            ymax = max(vals_ms) if vals_ms else 1.0
            ax.set_ylim(0, ymax * 2.5 if ymax > 0 else 1.0)

            # Make CCS23 bar more noticeable without changing data values.
            ccs_bar = bars[0]
            ccs_bar.set_hatch('////')
            ccs_bar.set_edgecolor('#111111')
            ccs_bar.set_linewidth(2.0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = _save_path('five_scheme_comparison')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def generate_avg(loaded):
    schemes = list(SCHEME_FOLDERS.keys())
    x = np.arange(len(schemes))
    w = 0.25

    phase_vals = {k: [] for k, _ in PHASES}
    for s in schemes:
        for pk, _ in PHASES:
            vals = [_mean_ms(loaded[s]['models'][mk], pk) for mk in MODEL_KEYS]
            phase_vals[pk].append(float(np.mean(vals)))

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle('Average Time by Scheme', fontsize=22, fontweight='bold')

    bars1 = ax.bar(
        x - w,
        phase_vals['encrypt_times'],
        width=w,
        color='none',
        edgecolor='#3A5A98',
        linewidth=1.4,
        hatch='/////',
        label='Encrypt',
    )
    bars2 = ax.bar(
        x,
        phase_vals['query_times'],
        width=w,
        color='none',
        edgecolor='#5FA8A3',
        linewidth=1.4,
        hatch='.....',
        label='Query',
    )
    bars3 = ax.bar(
        x + w,
        phase_vals['decrypt_times'],
        width=w,
        color='none',
        edgecolor='#E07A5F',
        linewidth=1.4,
        hatch='xxxxx',
        label='Decrypt',
    )

    ax.set_ylabel('Time (ms)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', alpha=0.28)
    ax.legend(fontsize=12, loc='upper right')

    # Keep values unchanged; tighten axis range so bars appear taller.
    ymax = max(
        max(phase_vals['encrypt_times']) if phase_vals['encrypt_times'] else 0.0,
        max(phase_vals['query_times']) if phase_vals['query_times'] else 0.0,
        max(phase_vals['decrypt_times']) if phase_vals['decrypt_times'] else 0.0,
    )
    ax.set_ylim(0.0, max(1.0, ymax * 1.35))

    # Make CCS23 bars easier to notice without changing values.
    for bars in [bars1, bars2, bars3]:
        ccs_bar = bars[0]
        ccs_bar.set_hatch('////')
        ccs_bar.set_edgecolor('#111111')
        ccs_bar.set_linewidth(2.0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = _save_path('five_scheme_avg_time')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def main():
    parser = argparse.ArgumentParser(description='Generate five-scheme figures with matched configs.')
    parser.add_argument('--num-records', type=int, default=128)
    parser.add_argument('--record-dim', type=int, default=128)
    parser.add_argument('--policy-size', type=int, default=10)
    parser.add_argument('--num-runs', type=int, default=3)
    args = parser.parse_args()

    loaded = {}
    sources = {}
    for scheme, folder in SCHEME_FOLDERS.items():
        path, data = _load_matched_result(
            folder,
            num_records=args.num_records,
            record_dim=args.record_dim,
            policy_size=args.policy_size,
            num_runs=args.num_runs,
        )
        if data is None:
            raise RuntimeError(
                f'No matched result for {scheme}: '
                f'num_records={args.num_records}, record_dim={args.record_dim}, '
                f'policy_size={args.policy_size}, num_runs={args.num_runs}'
            )
        loaded[scheme] = data
        sources[scheme] = os.path.basename(path)

    out_3x3 = generate_3x3(loaded)
    out_avg = generate_avg(loaded)

    print('Matched sources:')
    for scheme in SCHEME_FOLDERS:
        print(f'  {scheme}: {sources[scheme]}')
    print(f'3x3 figure: {out_3x3}')
    print(f'Avg figure: {out_avg}')


if __name__ == '__main__':
    main()
