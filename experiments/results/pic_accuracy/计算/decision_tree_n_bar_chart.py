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
    ('avg_encrypt_time', 'Encrypt'),
    ('avg_query_time', 'Query'),
    ('avg_decrypt_time', 'Decrypt'),
]
BROKEN_SCHEME = 'CCS23'


def compute_break_config(stage_data):
    positive_values = sorted(
        value
        for values in stage_data.values()
        for value in values
        if value > 0
    )
    if not positive_values:
        return None

    global_max = positive_values[-1]
    low_cluster = [value for value in positive_values if value <= global_max * 0.1]
    ccs_values = [value for value in stage_data.get(BROKEN_SCHEME, []) if value > 0]

    if low_cluster:
        lower_cap = max(low_cluster) * 1.2
    elif ccs_values:
        lower_cap = max(ccs_values) * 1.35
    else:
        lower_cap = positive_values[0] * 1.15

    upper_candidates = [value for value in positive_values if value > lower_cap]
    if not upper_candidates:
        return None

    upper_min = min(upper_candidates)
    upper_max = max(upper_candidates)
    if upper_min <= lower_cap * 1.6:
        return None

    return {
        'lower_min': 0.0,
        'lower_max': max(lower_cap, positive_values[0] * 1.05),
        'upper_min': upper_min * 0.92,
        'upper_max': upper_max * 1.08,
    }


def add_axis_break_marks(ax_top, ax_bottom):
    delta = 0.012
    kwargs = dict(color='#666', clip_on=False, linewidth=1.3)

    ax_top.plot((-delta, +delta), (-delta, +delta), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - delta, 1 + delta), (-delta, +delta), transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot((-delta, +delta), (1 - delta, 1 + delta), transform=ax_bottom.transAxes, **kwargs)
    ax_bottom.plot((1 - delta, 1 + delta), (1 - delta, 1 + delta), transform=ax_bottom.transAxes, **kwargs)


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


def plot_bar_chart(n_values, stage_curves, total_users, num_records, record_dim, policy_size, output_path):
    x = np.arange(len(n_values), dtype=float)
    bar_width = 0.16
    center = (len(SCHEMES) - 1) / 2.0

    fig = plt.figure(figsize=(15.2, 9.6))
    outer_grid = fig.add_gridspec(2, 2)
    legend_handles = []

    for index, (_, stage_label) in enumerate(STAGES):
        row = index // 2
        col = index % 2
        break_config = compute_break_config(stage_curves[stage_label])

        if break_config:
            inner_grid = outer_grid[row, col].subgridspec(2, 1, height_ratios=[3.3, 1.2], hspace=0.05)
            ax_top = fig.add_subplot(inner_grid[0])
            ax_bottom = fig.add_subplot(inner_grid[1], sharex=ax_top)
            plot_axes = [ax_top, ax_bottom]
        else:
            ax_top = fig.add_subplot(outer_grid[row, col])
            ax_bottom = ax_top
            plot_axes = [ax_top]

        for index, scheme in enumerate(SCHEMES):
            raw_values = stage_curves[stage_label][scheme]
            bars = None
            for ax in plot_axes:
                current_bars = ax.bar(
                    x + (index - center) * bar_width,
                    raw_values,
                    width=bar_width,
                    label=scheme,
                    color=FILL_COLORS[index],
                    edgecolor=SCHEME_COLORS[index],
                    linewidth=1.5,
                    hatch=HATCHES[index],
                    zorder=3,
                )
                if bars is None:
                    bars = current_bars
            if len(legend_handles) < len(SCHEMES):
                legend_handles.append(bars[0])

        ax_top.set_title(stage_label, fontsize=15)
        ax_bottom.set_xticks(x)
        ax_bottom.set_xticklabels([str(value) for value in n_values], fontsize=12)

        for ax in plot_axes:
            ax.grid(which='major', axis='y', linestyle='--', linewidth=1, color='#888', alpha=0.6, zorder=0)

        if break_config:
            ax_top.set_ylim(break_config['upper_min'], break_config['upper_max'])
            ax_bottom.set_ylim(break_config['lower_min'], break_config['lower_max'])
            ax_top.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)
            ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            add_axis_break_marks(ax_top, ax_bottom)
        else:
            all_values = [value for values in stage_curves[stage_label].values() for value in values]
            axis_max = max(all_values) if all_values else 1.0
            ax_top.set_ylim(0, max(axis_max * 1.08, 1.0))

        if col == 0:
            ax_top.set_ylabel('Latency (ms)', fontsize=14)

    fig.suptitle(f'Decision Tree Stage Latency vs n (N={total_users})', fontsize=17, y=0.97)
    fig.supxlabel('Block size n', fontsize=14, y=0.095)

    fig.legend(
        legend_handles,
        SCHEMES,
        fontsize=11,
        frameon=True,
        edgecolor='#ccc',
        loc='lower center',
        bbox_to_anchor=(0.5, 0.035),
        ncol=len(SCHEMES),
        columnspacing=1.2,
        handletextpad=0.6,
    )

    note = f'num_records={num_records}, record_dim={record_dim}, policy_size={policy_size}'
    fig.text(0.5, 0.008, note, ha='center', va='bottom', fontsize=10, color='#555')

    fig.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.2, wspace=0.18, hspace=0.38)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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
        default=str(PROJECT_ROOT / 'experiments' / 'results' / 'pic_accuracy' / '计算' / 'decision_tree_n_latency_bar.png'),
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
    plot_bar_chart(
        n_values=args.n_values,
        stage_curves=stage_curves,
        total_users=args.N,
        num_records=args.num_records,
        record_dim=args.record_dim,
        policy_size=args.policy_size,
        output_path=output_path,
    )

    print(f'Generated: {output_path}')
    print('Sources used:')
    for n_value in args.n_values:
        print(f'  n={n_value}')
        for scheme in SCHEMES:
            print(f'    {scheme}: {used_sources[n_value][scheme]}')


if __name__ == '__main__':
    main()