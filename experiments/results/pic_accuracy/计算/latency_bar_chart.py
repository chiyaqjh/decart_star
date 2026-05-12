import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = PROJECT_ROOT / 'experiments' / 'results'
COMPARE_DIR = PROJECT_ROOT / 'experiments' / 'compare'
if str(COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARE_DIR))

from accuracy_style import apply_accuracy_style

apply_accuracy_style()

SCHEMES = ['DeCart', 'DeCart*', 'CCS23', 'Server', 'Offline']
SCHEME_INFO = {
    'DeCart': 'our_decart',
    'DeCart*': 'our_decart_star',
    'CCS23': 'scheme1_ccs23',
    'Server': 'scheme2_server',
    'Offline': 'scheme3_offline',
}

SCHEME_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
FILL_COLORS = ['#90CAF9', '#FFE0B2', '#C8E6C9', '#E1BEE7', '#FFCDD2']
HATCHES = ['////', '...', 'xx', 'oo', '++']


def load_match(scheme_name: str, num_records: int, record_dim: int, policy_size: int, num_runs: int):
    folder = SCHEME_INFO[scheme_name]
    files = sorted(glob.glob(str(RESULTS_ROOT / folder / '*.json')), reverse=True)
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        cfg = data.get('config', {})
        if (
            cfg.get('num_records') == num_records
            and cfg.get('record_dim') == record_dim
            and cfg.get('policy_size') == policy_size
            and cfg.get('num_runs') == num_runs
        ):
            return data, Path(file_path).name
    return None, None


def aggregate_total_latency(data):
    model_dict = data.get('models', {})
    total_latency_ms = []
    for model_key in model_dict:
        model = model_dict[model_key]
        encrypt_ms = float(np.mean(np.array(model.get('encrypt_times', [0.0]), dtype=float))) * 1000.0
        query_ms = float(np.mean(np.array(model.get('query_times', [0.0]), dtype=float))) * 1000.0
        decrypt_ms = float(np.mean(np.array(model.get('decrypt_times', [0.0]), dtype=float))) * 1000.0
        total_latency_ms.append(encrypt_ms + query_ms + decrypt_ms)
    return float(np.mean(total_latency_ms)) if total_latency_ms else 0.0


def collect_latency_curve(n_values, fixed_n, policy_size, num_runs):
    curve = {scheme: [] for scheme in SCHEMES}
    used_sources = {}
    for num_records in n_values:
        used_sources[num_records] = {}
        for scheme in SCHEMES:
            data, source = load_match(scheme, num_records, fixed_n, policy_size, num_runs)
            if data is None:
                raise FileNotFoundError(
                    f'Missing result for {scheme} at num_records={num_records}, record_dim={fixed_n}, '
                    f'policy_size={policy_size}, num_runs={num_runs}'
                )
            curve[scheme].append(aggregate_total_latency(data))
            used_sources[num_records][scheme] = source
    return curve, used_sources


def plot_latency_bar(n_values, curve, output_path):
    x = np.arange(len(n_values), dtype=float)
    bar_width = 0.16
    center = (len(SCHEMES) - 1) / 2.0

    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    for index, scheme in enumerate(SCHEMES):
        ax.bar(
            x + (index - center) * bar_width,
            curve[scheme],
            width=bar_width,
            label=scheme,
            color=FILL_COLORS[index],
            edgecolor=SCHEME_COLORS[index],
            linewidth=1.5,
            hatch=HATCHES[index],
            zorder=3,
        )

    ax.set_title('Total Latency vs Total Users N', fontsize=16)
    ax.set_xlabel('Total number of users N', fontsize=14)
    ax.set_ylabel('Total Latency (ms)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in n_values], fontsize=12)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='y', linestyle='--', linewidth=1, color='#888', alpha=0.6, zorder=0)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.7, color='#bbb', alpha=0.4, zorder=0)
    ax.legend(fontsize=12, frameon=True, edgecolor='#ccc', loc='upper left')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate accuracy-style grouped bar chart for total latency vs N.')
    parser.add_argument('--fixed-n', type=int, default=16, help='Fixed record_dim for the N sweep.')
    parser.add_argument('--N-values', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    parser.add_argument('--policy-size', type=int, default=8)
    parser.add_argument('--num-runs', type=int, default=1)
    parser.add_argument(
        '--output',
        type=str,
        default=str(PROJECT_ROOT / 'experiments' / 'results' / 'pic_accuracy' / '计算' / 'scalability_fixed_n_time_comm_bar.png'),
    )
    args = parser.parse_args()

    curve, used_sources = collect_latency_curve(args.N_values, args.fixed_n, args.policy_size, args.num_runs)
    output_path = Path(args.output)
    plot_latency_bar(args.N_values, curve, output_path)

    print(f'Generated: {output_path}')
    print('Sources used:')
    for num_records in args.N_values:
        print(f'  N={num_records}, n={args.fixed_n}')
        for scheme in SCHEMES:
            print(f'    {scheme}: {used_sources[num_records][scheme]}')


if __name__ == '__main__':
    main()