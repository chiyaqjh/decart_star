"""
Additional paper figures generated from existing benchmark JSON files.

Figures:
1) Latency distribution boxplots (query/decrypt) for statistical stability.
2) Phase breakdown with error bars (mean ± std) for each model.
"""

import glob
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

from accuracy_style import apply_accuracy_style, get_pic_accuracy_dir, single_output_path

apply_accuracy_style()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / 'experiments' / 'results'
PIC_DIR = get_pic_accuracy_dir(PROJECT_ROOT)

SCHEMES = ['DeCart', 'DeCart*', 'Naive CCS-2023', 'Server', 'Offline', 'SecPQ']
FOLDERS = {
    'DeCart': 'our_decart',
    'DeCart*': 'our_decart_star',
    'Naive CCS-2023': 'naive_ccs23',
    'Server': 'scheme2_server',
    'Offline': 'scheme3_offline',
    'SecPQ': 'secpq',
}
MODELS = [('dot', 'Dot Product'), ('decision_tree', 'Decision Tree'), ('neural_network', 'Neural Network')]


def find_matched(folder: str, num_records: int, record_dim: int, policy_size: int, num_runs: int):
    files = sorted(glob.glob(str(RESULTS_ROOT / folder / '*.json')), reverse=True)
    for p in files:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = data.get('config', {})
        if (
            cfg.get('num_records') == num_records
            and cfg.get('record_dim') == record_dim
            and cfg.get('policy_size') == policy_size
            and cfg.get('num_runs') == num_runs
        ):
            return p, data
    raise RuntimeError(f'No matched file for {folder}')


def load_all(num_records=128, record_dim=128, policy_size=10, num_runs=3):
    loaded = {}
    sources = {}
    for s in SCHEMES:
        p, d = find_matched(FOLDERS[s], num_records, record_dim, policy_size, num_runs)
        loaded[s] = d
        sources[s] = os.path.basename(p)
    return loaded, sources


def ts_name(prefix: str):
    return single_output_path(PIC_DIR, prefix, 'png')


def plot_latency_boxplots(data):
    fig, axes = plt.subplots(2, 3, figsize=(16, 11.5), sharex=False)
    fig.suptitle('Latency Distribution by Scheme', fontsize=18, fontweight='bold')

    scheme_style = {
        'DeCart': {'edgecolor': '#3A5A98', 'hatch': '/////'},
        'DeCart*': {'edgecolor': '#5FA8A3', 'hatch': '.....'},
        'Naive CCS-2023': {'edgecolor': '#111111', 'hatch': '////'},
        'Server': {'edgecolor': '#E07A5F', 'hatch': 'xxxxx'},
        'Offline': {'edgecolor': '#6C757D', 'hatch': '++++'},
        'SecPQ': {'edgecolor': '#8C564B', 'hatch': '////+'},
    }

    for col, (mk, mlabel) in enumerate(MODELS):
        # query row
        axq = axes[0, col]
        q_samples = [np.array(data[s]['models'][mk].get('query_times', []), dtype=float) * 1000.0 for s in SCHEMES]
        q_bp = axq.boxplot(q_samples, tick_labels=SCHEMES, showfliers=False, patch_artist=True)
        for i, box in enumerate(q_bp['boxes']):
            style = scheme_style[SCHEMES[i]]
            box.set_facecolor('none')
            box.set_edgecolor(style['edgecolor'])
            box.set_hatch(style['hatch'])
            box.set_linewidth(2.0 if SCHEMES[i] == 'Naive CCS-2023' else 1.4)
            q_bp['medians'][i].set_color(style['edgecolor'])
            q_bp['medians'][i].set_linewidth(1.8 if SCHEMES[i] == 'Naive CCS-2023' else 1.4)
        axq.set_title(f'{mlabel} - Query')
        axq.tick_params(axis='x', rotation=15)
        axq.tick_params(axis='y', labelsize=10)
        axq.grid(alpha=0.25)

        # decrypt row
        axd = axes[1, col]
        d_samples = [np.array(data[s]['models'][mk].get('decrypt_times', []), dtype=float) * 1000.0 for s in SCHEMES]
        d_bp = axd.boxplot(d_samples, tick_labels=SCHEMES, showfliers=False, patch_artist=True)
        for i, box in enumerate(d_bp['boxes']):
            style = scheme_style[SCHEMES[i]]
            box.set_facecolor('none')
            box.set_edgecolor(style['edgecolor'])
            box.set_hatch(style['hatch'])
            box.set_linewidth(2.0 if SCHEMES[i] == 'Naive CCS-2023' else 1.4)
            d_bp['medians'][i].set_color(style['edgecolor'])
            d_bp['medians'][i].set_linewidth(1.8 if SCHEMES[i] == 'Naive CCS-2023' else 1.4)
        axd.set_title(f'{mlabel} - Decrypt')
        axd.tick_params(axis='x', rotation=15)
        axd.tick_params(axis='y', labelsize=10)
        axd.grid(alpha=0.25)

    axes[0, 0].set_ylabel('Query Time (ms)')
    axes[1, 0].set_ylabel('Decrypt Time (ms)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = ts_name('latency_boxplot')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_phase_errorbars(data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.6), sharey=False)
    fig.suptitle('Phase Latency Breakdown (Mean ± Std, ms)', fontsize=18, fontweight='bold')

    x = np.arange(len(SCHEMES))
    width = 0.25

    for ax, (mk, mlabel) in zip(axes, MODELS):
        e_mean = [np.mean(data[s]['models'][mk].get('encrypt_times', [0])) * 1000.0 for s in SCHEMES]
        q_mean = [np.mean(data[s]['models'][mk].get('query_times', [0])) * 1000.0 for s in SCHEMES]
        d_mean = [np.mean(data[s]['models'][mk].get('decrypt_times', [0])) * 1000.0 for s in SCHEMES]

        e_std = [np.std(data[s]['models'][mk].get('encrypt_times', [0])) * 1000.0 for s in SCHEMES]
        q_std = [np.std(data[s]['models'][mk].get('query_times', [0])) * 1000.0 for s in SCHEMES]
        d_std = [np.std(data[s]['models'][mk].get('decrypt_times', [0])) * 1000.0 for s in SCHEMES]

        ax.bar(
            x - width,
            e_mean,
            width,
            yerr=e_std,
            capsize=3,
            color='none',
            edgecolor='#3A5A98',
            linewidth=1.4,
            hatch='/////',
            label='Encrypt',
        )
        ax.bar(
            x,
            q_mean,
            width,
            yerr=q_std,
            capsize=3,
            color='none',
            edgecolor='#5FA8A3',
            linewidth=1.4,
            hatch='.....',
            label='Query',
        )
        ax.bar(
            x + width,
            d_mean,
            width,
            yerr=d_std,
            capsize=3,
            color='none',
            edgecolor='#E07A5F',
            linewidth=1.4,
            hatch='xxxxx',
            label='Decrypt',
        )

        ax.set_title(mlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(SCHEMES, rotation=15)
        ax.grid(axis='y', alpha=0.25)

    axes[0].set_ylabel('Latency (ms)')
    axes[0].legend(fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = ts_name('phase_errorbar')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def _pairwise_p_matrix(samples_by_scheme):
    n = len(SCHEMES)
    mat = np.ones((n, n), dtype=float)
    for i, si in enumerate(SCHEMES):
        xi = np.array(samples_by_scheme[si], dtype=float)
        for j, sj in enumerate(SCHEMES):
            if i == j:
                mat[i, j] = 1.0
                continue
            xj = np.array(samples_by_scheme[sj], dtype=float)
            if len(xi) < 2 or len(xj) < 2:
                mat[i, j] = np.nan
                continue
            p = ttest_ind(xi, xj, equal_var=False, nan_policy='omit').pvalue
            mat[i, j] = p
    return mat


def _p_to_star(p):
    if np.isnan(p):
        return 'NA'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def plot_significance_heatmap(data):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=False, sharey=False, constrained_layout=True)
    fig.suptitle('Pairwise Statistical Significance (Welch t-test p-values)', fontsize=14, fontweight='bold')

    phases = [('query_times', 'Query'), ('decrypt_times', 'Decrypt')]

    for r, (phase_key, phase_label) in enumerate(phases):
        for c, (mk, mlabel) in enumerate(MODELS):
            ax = axes[r, c]
            samples = {
                s: np.array(data[s]['models'][mk].get(phase_key, []), dtype=float) * 1000.0
                for s in SCHEMES
            }
            pmat = _pairwise_p_matrix(samples)

            # Display -log10(p) to emphasize very small p-values while keeping annotation as raw p.
            shown = -np.log10(np.clip(pmat, 1e-16, 1.0))
            im = ax.imshow(shown, cmap='YlOrRd', vmin=0.0, vmax=6.0)

            for i in range(len(SCHEMES)):
                for j in range(len(SCHEMES)):
                    p = pmat[i, j]
                    if i == j:
                        text = '1.0'
                    elif np.isnan(p):
                        text = 'NA'
                    elif p < 1e-3:
                        text = '<1e-3'
                    else:
                        text = f'{p:.3f}'
                    ax.text(j, i, text, ha='center', va='center', fontsize=7, color='black')

            ax.set_title(f'{mlabel} - {phase_label}')
            ax.set_xticks(range(len(SCHEMES)))
            ax.set_yticks(range(len(SCHEMES)))
            ax.set_xticklabels(SCHEMES, rotation=35, ha='right')
            ax.set_yticklabels(SCHEMES)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label('-log10(p-value)')

    out = ts_name('significance_heatmap')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def main():
    data, sources = load_all()
    out1 = plot_latency_boxplots(data)
    out2 = plot_phase_errorbars(data)
    out3 = plot_significance_heatmap(data)

    print('Matched sources:')
    for s in SCHEMES:
        print(f'  {s}: {sources[s]}')
    print(f'Generated: {out1}')
    print(f'Generated: {out2}')
    print(f'Generated: {out3}')


if __name__ == '__main__':
    main()
