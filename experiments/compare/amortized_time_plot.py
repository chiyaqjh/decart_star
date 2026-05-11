import glob
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from accuracy_style import apply_accuracy_style


ROOT = 'E:/decart/experiments/results'
PIC_DIR = 'E:/decart/experiments/results/pic_accuracy'

apply_accuracy_style()

SCHEME_FOLDERS = {
    'CCS23': 'scheme1_ccs23',
    'Server': 'scheme2_server',
    'Offline': 'scheme3_offline',
    'DeCart': 'our_decart',
    'DeCart*': 'our_decart_star',
}

MODEL_KEYS = ['dot', 'decision_tree', 'neural_network']


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
    return os.path.join(PIC_DIR, f'{prefix}.png')


def main():
    num_records = 128
    record_dim = 128
    policy_size = 10
    num_runs = 3

    loaded = {}
    sources = {}
    for scheme, folder in SCHEME_FOLDERS.items():
        path, data = _load_matched_result(folder, num_records, record_dim, policy_size, num_runs)
        if data is None:
            raise RuntimeError(
                f'No matched result for {scheme}: '
                f'num_records={num_records}, record_dim={record_dim}, policy_size={policy_size}, num_runs={num_runs}'
            )
        loaded[scheme] = data
        sources[scheme] = os.path.basename(path)

    schemes = list(SCHEME_FOLDERS.keys())
    enc = []
    qry = []
    dec = []

    for s in schemes:
        e_vals = [_mean_ms(loaded[s]['models'][mk], 'encrypt_times') for mk in MODEL_KEYS]
        q_vals = [_mean_ms(loaded[s]['models'][mk], 'query_times') for mk in MODEL_KEYS]
        d_vals = [_mean_ms(loaded[s]['models'][mk], 'decrypt_times') for mk in MODEL_KEYS]

        enc.append(float(np.mean(e_vals)))
        qry.append(float(np.mean(q_vals)))
        dec.append(float(np.mean(d_vals)))

    amort_enc = [v / max(num_runs, 1) for v in enc]
    amort_total = [a + q + d for a, q, d in zip(amort_enc, qry, dec)]
    raw_total = [e + q + d for e, q, d in zip(enc, qry, dec)]

    x = np.arange(len(schemes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(13, 8.6))
    fig.suptitle('Amortized End-to-End Time by Scheme', fontsize=19, fontweight='bold')

    bars_raw = ax.bar(
        x - w / 2,
        raw_total,
        width=w,
        color='none',
        edgecolor='#3A5A98',
        linewidth=1.6,
        hatch='/////',
        label='Raw = Encrypt + Query + Decrypt',
    )
    bars_amort = ax.bar(
        x + w / 2,
        amort_total,
        width=w,
        color='none',
        edgecolor='#E07A5F',
        linewidth=1.6,
        hatch='.....',
        label='Amortized = Encrypt/num_runs + Query + Decrypt',
    )

    ax.set_xticks(x)
    ax.set_xticklabels(schemes, fontsize=14)
    ax.set_ylabel('Time (ms)', fontsize=15)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=12, loc='upper right')

    # Highlight CCS23 for visibility (same style idea as previous chart updates)
    bars_raw[0].set_hatch('////')
    bars_raw[0].set_linewidth(2.2)
    bars_amort[0].set_hatch('////')
    bars_amort[0].set_linewidth(2.2)

    ax.set_ylim(0.0, max(raw_total) * 1.35 if raw_total else 1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out = _save_path('five_scheme_amortized_time')
    plt.savefig(out, dpi=170, bbox_inches='tight')
    plt.close()

    print('Matched sources:')
    for scheme in SCHEME_FOLDERS:
        print(f'  {scheme}: {sources[scheme]}')
    print(f'Output: {out}')


if __name__ == '__main__':
    main()
