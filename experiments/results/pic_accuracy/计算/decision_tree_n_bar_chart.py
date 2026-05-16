import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3]
RESULTS_ROOT = PROJECT_ROOT / 'experiments' / 'results'
COMPARE_DIR = PROJECT_ROOT / 'experiments' / 'compare'
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'results' / 'pic_accuracy' / '计算_computation'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARE_DIR))

from config import Config
from accuracy_style import apply_accuracy_style


apply_accuracy_style()

SCHEMES = ['DeCart', 'DeCart*', 'CCS23', 'Server', 'Offline']
SCHEME_INFO = {
    'DeCart': 'our_decart/decart_multimodel_exp_*.json',
    'DeCart*': 'our_decart_star/decart_star_multimodel_exp_*.json',
    'CCS23': 'scheme1_ccs23/ccs23_exp_*.json',
    'Server': 'scheme2_server/server_exp_*.json',
    'Offline': 'scheme3_offline/offline_exp_*.json',
}
SCHEME_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
FILL_COLORS = ['#90CAF9', '#FFE0B2', '#C8E6C9', '#E1BEE7', '#FFCDD2']
HATCHES = ['////', '...', 'xx', 'oo', '++']
STAGES = [
    ('avg_setup_time', 'Setup'),
    ('avg_keygen_time', 'KeyGen'),
    ('avg_register_time', 'Register'),
    ('avg_encrypt_time', 'Encrypt'),
    ('avg_query_time', 'Query'),
    ('avg_decrypt_time', 'Decrypt'),
]
DECART_ONLY_STAGES = {'Setup', 'KeyGen', 'Register'}


def schemes_for_stage(stage_label):
    if stage_label in DECART_ONLY_STAGES:
        return ['DeCart', 'DeCart*']
    return SCHEMES


def load_latest_match(scheme_name, n_value, total_users, num_records, record_dim, policy_size, num_runs, model_key):
    pattern = RESULTS_ROOT / SCHEME_INFO[scheme_name]
    file_paths = sorted(glob.glob(str(pattern)), key=lambda path: Path(path).stat().st_mtime, reverse=True)
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
        except Exception:
            continue

        config = data.get('config', {})
        summary = data.get('summary', {}).get(model_key)
        if not summary:
            continue

        if (
            config.get('N') == total_users
            and config.get('n') == n_value
            and config.get('num_records') == num_records
            and config.get('record_dim') == record_dim
            and config.get('policy_size') == policy_size
            and config.get('num_runs') == num_runs
        ):
            return data, Path(file_path).name

    raise FileNotFoundError(
        f'Missing {scheme_name} result for N={total_users}, n={n_value}, '
        f'num_records={num_records}, record_dim={record_dim}, policy_size={policy_size}, num_runs={num_runs}'
    )


def collect_stage_curves(n_values, total_users, num_records, record_dim, policy_size, num_runs, model_key):
    stage_curves = {stage_label: {scheme: [] for scheme in SCHEMES} for _, stage_label in STAGES}
    used_sources = {n_value: {} for n_value in n_values}
    for n_value in n_values:
        for scheme in SCHEMES:
            data, source = load_latest_match(
                scheme,
                n_value,
                total_users,
                num_records,
                record_dim,
                policy_size,
                num_runs,
                model_key,
            )
            summary = data['summary'][model_key]
            for summary_key, stage_label in STAGES:
                stage_curves[stage_label][scheme].append(float(summary.get(summary_key, 0.0)) * 1000.0)
            used_sources[n_value][scheme] = source
    return stage_curves, used_sources


def _slugify_stage_label(stage_label):
    return stage_label.lower().replace(' ', '_')


def plot_single_stage_chart(n_values, stage_label, stage_values, total_users, num_records, record_dim, policy_size, output_path):
    stage_schemes = schemes_for_stage(stage_label)
    x = np.arange(len(n_values), dtype=float)
    bar_width = 0.26 if len(stage_schemes) == 2 else 0.16
    center = (len(stage_schemes) - 1) / 2.0
    legend_handles = []

    fig = plt.figure(figsize=(9.8, 6.8))
    ax_top = fig.add_subplot(111)

    for index, scheme in enumerate(stage_schemes):
        raw_values = stage_values[scheme]
        bars = ax_top.bar(
            x + (index - center) * bar_width,
            raw_values,
            width=bar_width,
            label=scheme,
            color=FILL_COLORS[SCHEMES.index(scheme)],
            edgecolor=SCHEME_COLORS[SCHEMES.index(scheme)],
            linewidth=1.5,
            hatch=HATCHES[SCHEMES.index(scheme)],
            zorder=3,
        )
        legend_handles.append(bars[0])

    ax_top.set_xticks(x)
    ax_top.set_xticklabels([str(value) for value in n_values], fontsize=12)
    ax_top.set_xlabel('Block size n', fontsize=14)
    ax_top.set_ylabel('Latency (ms)', fontsize=14)

    ax_top.grid(which='major', axis='y', linestyle='--', linewidth=1, color='#888', alpha=0.6, zorder=0)
    all_values = [value for values in stage_values.values() for value in values]
    axis_max = max(all_values) if all_values else 1.0
    ax_top.set_ylim(0, max(axis_max * 1.08, 1.0))

    ax_top.text(
        0.03,
        0.97,
        f'{stage_label}\nN={total_users}',
        transform=ax_top.transAxes,
        ha='left',
        va='top',
        fontsize=14,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9),
        zorder=4,
    )

    ax_top.legend(
        legend_handles,
        stage_schemes,
        fontsize=10,
        frameon=True,
        edgecolor='#ccc',
        loc='upper left',
        bbox_to_anchor=(0.02, 0.78),
        ncol=1,
    )

    note = f'num_records={num_records}, record_dim={record_dim}, policy_size={policy_size}'
    fig.text(
        0.97,
        0.08,
        note,
        ha='right',
        va='bottom',
        fontsize=10,
        color='#555',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#dddddd', alpha=0.85),
    )

    fig.subplots_adjust(left=0.1, right=0.96, top=0.88, bottom=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_stage_charts(n_values, stage_curves, total_users, num_records, record_dim, policy_size, output_path):
    output_dir = output_path.parent
    output_stem = output_path.stem
    for _, stage_label in STAGES:
        stage_output = output_dir / f'{output_stem}_{_slugify_stage_label(stage_label)}.png'
        plot_single_stage_chart(
            n_values=n_values,
            stage_label=stage_label,
            stage_values=stage_curves[stage_label],
            total_users=total_users,
            num_records=num_records,
            record_dim=record_dim,
            policy_size=policy_size,
            output_path=stage_output,
        )


def main():
    parser = argparse.ArgumentParser(description='Generate grouped bar charts for decision-tree stage latency vs n.')
    parser.add_argument('--N', type=int, default=Config.MAX_USERS, help='Total users N.')
    parser.add_argument('--n-values', type=int, nargs='+', default=[16, 32, 64, 128, 256], help='n values to compare.')
    parser.add_argument('--num-records', type=int, default=10, help='Number of records used to filter results.')
    parser.add_argument('--record-dim', type=int, default=10, help='Record dimension used to filter results.')
    parser.add_argument('--policy-size', type=int, default=Config.EXPERIMENT_POLICY_SIZE, help='Policy size used to filter results.')
    parser.add_argument('--num-runs', type=int, default=1, help='Run count used to filter results.')
    parser.add_argument('--model-key', type=str, default='decision_tree', help='Model key in result summaries.')
    parser.add_argument(
        '--output',
        type=str,
        default=str(OUTPUT_DIR / 'decision_tree_n_latency.png'),
    )
    args = parser.parse_args()

    stage_curves, used_sources = collect_stage_curves(
        n_values=args.n_values,
        total_users=args.N,
        num_records=args.num_records,
        record_dim=args.record_dim,
        policy_size=args.policy_size,
        num_runs=args.num_runs,
        model_key=args.model_key,
    )
    output_path = Path(args.output)
    plot_stage_charts(
        n_values=args.n_values,
        stage_curves=stage_curves,
        total_users=args.N,
        num_records=args.num_records,
        record_dim=args.record_dim,
        policy_size=args.policy_size,
        output_path=output_path,
    )

    print(f'Generated charts under: {output_path.parent}')
    print('Sources used:')
    for n_value in args.n_values:
        print(f'  n={n_value}')
        for scheme in SCHEMES:
            print(f'    {scheme}: {used_sources[n_value][scheme]}')


if __name__ == '__main__':
    main()