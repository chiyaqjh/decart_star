import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogFormatterMathtext


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3]
RESULTS_ROOT = PROJECT_ROOT / 'experiments' / 'results'
DATA_ROOT = RESULTS_ROOT / 'data_new'
COMPARE_DIR = PROJECT_ROOT / 'experiments' / 'compare'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARE_DIR))

from accuracy_style import apply_accuracy_style


apply_accuracy_style()

OUT_DIR = RESULTS_ROOT / 'pic_new' / 'communication'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEME_STYLES = {
    'SecPQ': {'edgecolor': '#2E7D32', 'facecolor': '#C8E6C9', 'hatch': '\\'},
    'plaintext': {'edgecolor': '#1B5E20', 'facecolor': '#DCEDC8', 'hatch': '..'},
    'DeCart': {'edgecolor': '#2196F3', 'facecolor': '#90CAF9', 'hatch': '//'},
    'DeCart*': {'edgecolor': '#FF9800', 'facecolor': '#FFE0B2', 'hatch': 'xx'},
    'Server': {'edgecolor': '#9C27B0', 'facecolor': '#E1BEE7', 'hatch': 'oo'},
    'Offline': {'edgecolor': '#F44336', 'facecolor': '#FFCDD2', 'hatch': '++'},
}

MODEL_CONFIGS = {
    'decision_tree': {
        'output_dir': OUT_DIR,
        'schemes': ['SecPQ', 'plaintext', 'DeCart', 'DeCart*', 'Server', 'Offline'],
        'folders': {
            'SecPQ': [DATA_ROOT / 'secpq'],
            'plaintext': [DATA_ROOT / 'scheme1_ccs23'],
            'DeCart': [DATA_ROOT / 'our_decart'],
            'DeCart*': [DATA_ROOT / 'our_decart_star'],
            'Server': [DATA_ROOT / 'scheme2_server'],
            'Offline': [DATA_ROOT / 'scheme3_offline'],
        },
    },
    'dot': {
        'output_dir': OUT_DIR / 'dot',
        'schemes': ['plaintext', 'DeCart', 'DeCart*', 'Server', 'Offline'],
        'folders': {
            'plaintext': [DATA_ROOT / 'scheme1_ccs23' / 'dot_N10000_n32' / 'q=1'],
            'DeCart': [DATA_ROOT / 'our_decart' / 'dot_N10000_n32' / 'q=1'],
            'DeCart*': [DATA_ROOT / 'our_decart_star' / 'dot_N10000_n32' / 'q=1'],
            'Server': [DATA_ROOT / 'scheme2_server' / 'dot_N10000_n32' / 'q=1'],
            'Offline': [DATA_ROOT / 'scheme3_offline' / 'dot_N10000_n32' / 'q=1'],
        },
    },
    'neural_network': {
        'output_dir': OUT_DIR / 'neural_network',
        'schemes': ['plaintext', 'DeCart', 'DeCart*', 'Server', 'Offline'],
        'folders': {
            'plaintext': [DATA_ROOT / 'scheme1_ccs23' / 'neural_network_N10000_n32' / 'q=1'],
            'DeCart': [DATA_ROOT / 'our_decart' / 'neural_network_N10000_n32' / 'q=1'],
            'DeCart*': [DATA_ROOT / 'our_decart_star' / 'neural_network_N10000_n32' / 'q=1'],
            'Server': [DATA_ROOT / 'scheme2_server' / 'neural_network_N10000_n32' / 'q=1'],
            'Offline': [DATA_ROOT / 'scheme3_offline' / 'neural_network_N10000_n32' / 'q=1'],
        },
    },
}

FIXED_N = 10000
FIXED_BLOCK_SIZE = 32
FIXED_POLICY_SIZE = 32
FIXED_NUM_RUNS = 1
RESULT_CACHE = {}


def estimate_metric_from_q1(model_name, scheme_name, num_records, record_dim, num_queriers, metric):
    """Estimate missing q>1 communication from q=1: one upload + repeated check/query/decrypt."""
    if num_queriers <= 1:
        return float('nan')

    data_q1 = load_match(model_name, scheme_name, num_records, record_dim, 1)
    if data_q1 is None:
        return float('nan')

    upload = metric_for_scheme(data_q1, model_name, 'upload_kb')
    check = metric_for_scheme(data_q1, model_name, 'check_kb')
    query = metric_for_scheme(data_q1, model_name, 'query_kb')
    decrypt = metric_for_scheme(data_q1, model_name, 'decrypt_kb')

    if metric == 'upload_kb':
        return upload
    if metric == 'check_kb':
        return check * num_queriers
    if metric == 'query_kb':
        return query * num_queriers
    if metric == 'decrypt_kb':
        return decrypt * num_queriers
    if metric == 'total_kb':
        parts = [upload, check, query, decrypt]
        if any(np.isnan(part) for part in parts):
            return float('nan')
        return upload + (check + query + decrypt) * num_queriers

    return float('nan')


def resolve_folders(model_name, scheme_name, num_queriers):
    folders = list(MODEL_CONFIGS[model_name]['folders'][scheme_name])
    if model_name in {'dot', 'neural_network'} and num_queriers != 1:
        resolved = []
        for folder in folders:
            if folder.name.startswith('q='):
                resolved.append(folder.parent / f'q={num_queriers}')
            else:
                resolved.append(folder)
        return resolved
    return folders

METRIC_SPECS = [
    ('upload_kb', 'encrypt_kb', 'Communication cost (KB)'),
    ('check_kb', 'check_kb', 'Communication cost (KB)'),
    ('query_kb', 'query_kb', 'Communication cost (KB)'),
    ('decrypt_kb', 'decrypt_kb', 'Communication cost (KB)'),
    ('total_kb', 'total_kb', 'Communication cost (KB)'),
]


def style_axes(ax):
    if ax.get_yscale() == 'linear':
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='y', linestyle='--', linewidth=1, color='#888', alpha=0.6, zorder=0)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.7, color='#bbb', alpha=0.4, zorder=0)


def _mean_kb(values):
    if not values:
        return float('nan')
    arr = np.array(values, dtype=float)
    return float(np.mean(arr) / 1024.0)


def _sum_kb(values):
    if not values:
        return float('nan')
    arr = np.array(values, dtype=float)
    return float(np.sum(arr) / 1024.0)


def _mean_seconds(values):
    if not values:
        return float('nan')
    arr = np.array(values, dtype=float)
    return float(np.mean(arr))


def _phase_comm_kb(model_block):
    upload_sizes = model_block.get('comm_upload_sizes')
    check_sizes = model_block.get('comm_check_sizes')
    query_sizes = model_block.get('comm_query_sizes')
    decrypt_sizes = model_block.get('comm_decrypt_sizes')

    # Some older baseline JSONs exported upload/query/decrypt explicitly but omitted
    # comm_check_sizes because check is semantically zero. Treat that case as a real
    # phase split instead of falling back to time-based proportional allocation.
    if upload_sizes and query_sizes and decrypt_sizes:
        upload_kb = _mean_kb(model_block.get('comm_upload_sizes'))
        check_kb = _mean_kb(check_sizes) if check_sizes is not None else 0.0
        query_kb = _mean_kb(model_block.get('comm_query_sizes'))
        decrypt_kb = _mean_kb(model_block.get('comm_decrypt_sizes'))
        return {
            'upload_kb': upload_kb,
            'check_kb': check_kb,
            'query_kb': query_kb,
            'decrypt_kb': decrypt_kb,
            'total_kb': upload_kb + check_kb + query_kb + decrypt_kb,
        }

    total_kb = _sum_kb(model_block.get('communication_sizes'))
    if np.isnan(total_kb):
        return {
            'upload_kb': float('nan'),
            'check_kb': float('nan'),
            'query_kb': float('nan'),
            'decrypt_kb': float('nan'),
            'total_kb': float('nan'),
        }

    encrypt_s = max(_mean_seconds(model_block.get('encrypt_times')), 0.0)
    query_s = max(_mean_seconds(model_block.get('query_times')), 0.0)
    decrypt_s = max(_mean_seconds(model_block.get('decrypt_times')), 0.0)
    total_s = encrypt_s + query_s + decrypt_s
    if total_s <= 0:
        ratios = {
            'upload_kb': 1.0 / 3.0,
            'query_kb': 1.0 / 3.0,
            'decrypt_kb': 1.0 / 3.0,
        }
    else:
        ratios = {
            'upload_kb': encrypt_s / total_s,
            'query_kb': query_s / total_s,
            'decrypt_kb': decrypt_s / total_s,
        }

    return {
        'upload_kb': total_kb * ratios['upload_kb'],
        'check_kb': 0.0,
        'query_kb': total_kb * ratios['query_kb'],
        'decrypt_kb': total_kb * ratios['decrypt_kb'],
        'total_kb': total_kb,
    }


def metric_for_scheme(data, model_name, metric):
    if data is None:
        return float('nan')

    model_block = data.get('models', {}).get(model_name, {})
    if not model_block:
        return float('nan')

    return _phase_comm_kb(model_block).get(metric, float('nan'))


def load_match(model_name, scheme_name, num_records, record_dim, num_queriers):
    cache_key = (model_name, scheme_name, num_records, record_dim, num_queriers)
    if cache_key in RESULT_CACHE:
        return RESULT_CACHE[cache_key]

    folders = resolve_folders(model_name, scheme_name, num_queriers)
    for folder in folders:
        files = sorted(glob.glob(str(folder / '*.json')), reverse=True)
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            cfg = data.get('config', {})
            model_types = cfg.get('model_types') or list((data.get('models') or {}).keys())
            if len(model_types) != 1 or model_types[0] != model_name:
                continue
            if (
                cfg.get('N') == FIXED_N
                and cfg.get('n') == FIXED_BLOCK_SIZE
                and cfg.get('num_records') == num_records
                and cfg.get('record_dim') == record_dim
                and cfg.get('policy_size') == FIXED_POLICY_SIZE
                and cfg.get('num_runs') == FIXED_NUM_RUNS
                and cfg.get('num_queriers', 1) == num_queriers
            ):
                RESULT_CACHE[cache_key] = data
                return data

    RESULT_CACHE[cache_key] = None
    return None


def x_values_for_querier(num_queriers):
    return [10, 100, 500, 1000, 5000, 10000]


def curve_for_metric(model_name, metric, num_queriers):
    schemes = MODEL_CONFIGS[model_name]['schemes']
    x_values = x_values_for_querier(num_queriers)
    curve = {scheme: [] for scheme in schemes}
    for value in x_values:
        for scheme in schemes:
            data = load_match(model_name, scheme, num_records=value, record_dim=value, num_queriers=num_queriers)
            if data is not None:
                curve[scheme].append(metric_for_scheme(data, model_name, metric))
            else:
                curve[scheme].append(
                    estimate_metric_from_q1(
                        model_name,
                        scheme,
                        num_records=value,
                        record_dim=value,
                        num_queriers=num_queriers,
                        metric=metric,
                    )
                )
    return x_values, curve


def save_figure(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Generated: {output_path}')


def plot_bar_chart(model_name, metric, output_name, y_label, num_queriers, out_dir):
    schemes = MODEL_CONFIGS[model_name]['schemes']
    x_values, curve = curve_for_metric(model_name, metric, num_queriers)
    categories = [str(v) for v in x_values]
    x = np.arange(len(categories), dtype=float)
    bar_width = 0.11
    center = (len(schemes) - 1) / 2.0

    positive_values = [
        value
        for scheme_values in curve.values()
        for value in scheme_values
        if np.isfinite(value) and value > 0
    ]
    floor_value = min(positive_values) / 5.0 if positive_values else 1e-3
    sanitized = {
        scheme: [value if np.isfinite(value) and value > 0 else floor_value for value in values]
        for scheme, values in curve.items()
    }

    fig, ax = plt.subplots(figsize=(11.0, 7.4))
    bar_patches = []
    for idx, scheme in enumerate(schemes):
        style = SCHEME_STYLES[scheme]
        container = ax.bar(
            x + (idx - center) * bar_width,
            sanitized[scheme],
            width=bar_width,
            label=scheme,
            color=style['facecolor'],
            edgecolor=style['edgecolor'],
            linewidth=1.4,
            hatch=style['hatch'],
            zorder=3,
        )
        bar_patches.extend(container.patches)
    ax.set_yscale('log')
    ax.set_xlabel('Number of data records', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    if metric == 'query_kb':
        ax.set_ylim(bottom=8e2, top=1e5)
        ax.set_yticks([1e3, 1e4, 1e5])
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    style_axes(ax)
    legend = ax.legend(frameon=True, edgecolor='#ccc')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer)
    overlaps_bars = any(
        legend_bbox.overlaps(patch.get_window_extent(renderer=renderer))
        for patch in bar_patches
        if patch.get_height() > 0
    )

    if overlaps_bars:
        legend.remove()
        fig.set_size_inches(11.0, 8.6, forward=True)
        ax.legend(frameon=True, edgecolor='#ccc')

    save_figure(fig, out_dir / f'{output_name}.png')


def generate_for_querier(model_name, num_queriers):
    out_dir = MODEL_CONFIGS[model_name]['output_dir'] / f'querier_{num_queriers}'
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, output_name, y_label in METRIC_SPECS:
        plot_bar_chart(model_name, metric, output_name, y_label, num_queriers, out_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate communication charts from data_new results.')
    parser.add_argument(
        '--models',
        nargs='*',
        choices=sorted(MODEL_CONFIGS.keys()),
        default=['decision_tree'],
        help='Model names to generate.',
    )
    parser.add_argument('--queriers', type=int, nargs='*', default=[1, 10, 50, 100], help='Querier counts to plot.')
    args = parser.parse_args()

    for model_name in args.models:
        RESULT_CACHE.clear()
        for num_queriers in args.queriers:
            generate_for_querier(model_name, num_queriers)

    print(f'All communication charts are saved in: {OUT_DIR}')


if __name__ == '__main__':
    main()