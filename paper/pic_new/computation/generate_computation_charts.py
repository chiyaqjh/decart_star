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
DATA_ROOT = RESULTS_ROOT / 'data_new'
COMPARE_DIR = PROJECT_ROOT / 'experiments' / 'compare'
if str(COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARE_DIR))

from accuracy_style import apply_accuracy_style, style_axes


apply_accuracy_style()

OUT_ROOT = RESULTS_ROOT / 'pic_new' / 'computation'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_SIZES = [10, 100, 500, 1000, 5000, 10000]

SCHEME_STYLES = {
    'SecPQ': {'edgecolor': '#2E7D32', 'facecolor': '#C8E6C9', 'hatch': '\\'},
    'Naive_ccs23': {'edgecolor': '#66BB6A', 'facecolor': '#E8F5E9', 'hatch': '--'},
    'plaintext': {'edgecolor': '#1B5E20', 'facecolor': '#DCEDC8', 'hatch': '..'},
    'DeCart': {'edgecolor': '#2196F3', 'facecolor': '#90CAF9', 'hatch': '//'},
    'DeCart*': {'edgecolor': '#FF9800', 'facecolor': '#FFE0B2', 'hatch': 'xx'},
    'Server': {'edgecolor': '#9C27B0', 'facecolor': '#E1BEE7', 'hatch': 'oo'},
    'Offline': {'edgecolor': '#F44336', 'facecolor': '#FFCDD2', 'hatch': '++'},
}

MODEL_CONFIGS = {
    'decision_tree': {
        'output_dir': OUT_ROOT,
        'schemes': ['SecPQ', 'Naive_ccs23', 'plaintext', 'DeCart', 'DeCart*', 'Server', 'Offline'],
        'folders': {
            'SecPQ': [DATA_ROOT / 'secpq'],
            'Naive_ccs23': [DATA_ROOT / 'naive_ccs23'],
            'plaintext': [DATA_ROOT / 'scheme1_ccs23'],
            'DeCart': [
                DATA_ROOT / 'our_decart' / 'decision_tree' / 'q=1',
                DATA_ROOT / 'our_decart',
            ],
            'DeCart*': [
                DATA_ROOT / 'our_decart_star' / 'decision_tree' / 'q=1',
                DATA_ROOT / 'our_decart_star',
            ],
            'Server': [DATA_ROOT / 'scheme2_server'],
            'Offline': [DATA_ROOT / 'scheme3_offline'],
        },
    },
    'dot': {
        'output_dir': OUT_ROOT / 'dot',
        'schemes': ['Naive_ccs23', 'plaintext', 'DeCart', 'DeCart*', 'Server', 'Offline'],
        'folders': {
            'Naive_ccs23': [DATA_ROOT / 'naive_ccs23' / 'dot_N10000_n32'],
            'plaintext': [DATA_ROOT / 'scheme1_ccs23' / 'dot_N10000_n32'],
            'DeCart': [DATA_ROOT / 'our_decart' / 'dot_N10000_n32'],
            'DeCart*': [DATA_ROOT / 'our_decart_star' / 'dot_N10000_n32'],
            'Server': [DATA_ROOT / 'scheme2_server' / 'dot_N10000_n32'],
            'Offline': [DATA_ROOT / 'scheme3_offline' / 'dot_N10000_n32'],
        },
    },
    'neural_network': {
        'output_dir': OUT_ROOT / 'neural_network',
        'schemes': ['Naive_ccs23', 'plaintext', 'DeCart', 'DeCart*', 'Server', 'Offline'],
        'folders': {
            'Naive_ccs23': [DATA_ROOT / 'naive_ccs23' / 'neural_network_N10000_n32'],
            'plaintext': [DATA_ROOT / 'scheme1_ccs23' / 'neural_network_N10000_n32'],
            'DeCart': [DATA_ROOT / 'our_decart' / 'neural_network_N10000_n32'],
            'DeCart*': [DATA_ROOT / 'our_decart_star' / 'neural_network_N10000_n32'],
            'Server': [DATA_ROOT / 'scheme2_server' / 'neural_network_N10000_n32'],
            'Offline': [DATA_ROOT / 'scheme3_offline' / 'neural_network_N10000_n32'],
        },
    },
}

METRIC_SPECS = [
    ('check_ms', 'check_ms.png', 'Running time (ms)'),
    ('encrypt_ms', 'encrypt_ms.png', 'Running time (ms)'),
    ('query_ms', 'query_ms.png', 'Running time (ms)'),
    ('decrypt_ms', 'decrypt_ms.png', 'Running time (ms)'),
    ('total_ms', 'total_ms.png', 'Running time (ms)'),
]


def _first_or_nan(values):
    if not values:
        return float('nan')
    return float(values[0])


def _summary_ms(summary_block, key):
    value = summary_block.get(key)
    if value is None:
        return float('nan')
    return float(value) * 1000.0


def metric_for_model(doc, model_name, metric):
    model_block = (doc.get('models') or {}).get(model_name, {})
    summary_block = (doc.get('summary') or {}).get(model_name, {})

    if metric == 'check_ms':
        value = _summary_ms(summary_block, 'avg_check_time')
        if not np.isnan(value):
            return value
        return _first_or_nan(model_block.get('check_times', [])) * 1000.0

    if metric == 'encrypt_ms':
        value = _summary_ms(summary_block, 'avg_encrypt_time')
        if not np.isnan(value):
            return value
        return _first_or_nan(model_block.get('encrypt_times', [])) * 1000.0

    if metric == 'query_ms':
        value = _summary_ms(summary_block, 'avg_query_time')
        if not np.isnan(value):
            return value
        return _first_or_nan(model_block.get('query_times', [])) * 1000.0

    if metric == 'decrypt_ms':
        value = _summary_ms(summary_block, 'avg_decrypt_time')
        if not np.isnan(value):
            return value
        return _first_or_nan(model_block.get('decrypt_times', [])) * 1000.0

    if metric == 'total_ms':
        parts = [
            metric_for_model(doc, model_name, 'encrypt_ms'),
            metric_for_model(doc, model_name, 'query_ms'),
            metric_for_model(doc, model_name, 'decrypt_ms'),
        ]
        if any(np.isnan(part) for part in parts):
            return float('nan')
        return float(sum(parts))

    raise ValueError(f'Unsupported metric: {metric}')


def resolve_folders(model_name, scheme, num_queriers):
    folders = list(MODEL_CONFIGS[model_name]['folders'][scheme])
    if model_name == 'decision_tree' and num_queriers == 100:
        if scheme == 'DeCart':
            return [DATA_ROOT / 'our_decart' / 'decision_tree' / 'q=100']
        if scheme == 'DeCart*':
            return [DATA_ROOT / 'our_decart_star' / 'decision_tree' / 'q=100']
    return folders


def load_latest_result(model_name, scheme, size, num_queriers):
    folders = resolve_folders(model_name, scheme, num_queriers)
    for folder in folders:
        candidates = []
        for path_str in glob.glob(str(folder / '*.json')):
            path = Path(path_str)
            with open(path, 'r', encoding='utf-8') as handle:
                doc = json.load(handle)
            cfg = doc.get('config', {})
            model_types = cfg.get('model_types') or list((doc.get('models') or {}).keys())
            if len(model_types) != 1 or model_types[0] != model_name:
                continue
            if cfg.get('num_records') != size or cfg.get('record_dim') != size:
                continue
            if cfg.get('num_queriers') != num_queriers:
                continue
            candidates.append((path.name, doc))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[-1][1]
    return None


def collect_metric(model_name, metric, num_queriers, sizes):
    schemes = MODEL_CONFIGS[model_name]['schemes']
    values_map = {scheme: [] for scheme in schemes}
    for size in sizes:
        for scheme in schemes:
            doc = load_latest_result(model_name, scheme, size, num_queriers)
            if doc is None:
                values_map[scheme].append(float('nan'))
            else:
                values_map[scheme].append(metric_for_model(doc, model_name, metric))
    return values_map


def plot_metric(model_name, metric, output_name, y_label, num_queriers, out_dir, sizes):
    schemes = MODEL_CONFIGS[model_name]['schemes']
    values_map = collect_metric(model_name, metric, num_queriers, sizes)
    x = np.arange(len(sizes), dtype=float)
    width = 0.12
    center = (len(schemes) - 1) / 2.0

    fig, ax = plt.subplots(figsize=(11.6, 7.8))
    for idx, scheme in enumerate(schemes):
        heights = np.array(values_map[scheme], dtype=float)
        style = SCHEME_STYLES[scheme]
        ax.bar(
            x + (idx - center) * width,
            heights,
            width=width,
            label=scheme,
            color=style['facecolor'],
            edgecolor=style['edgecolor'],
            linewidth=1.3,
            hatch=style['hatch'],
            zorder=3,
        )

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel('Number of data records')
    ax.set_ylabel(y_label)
    style_axes(ax)
    ax.legend(frameon=True, edgecolor='#ccc')

    fig.tight_layout()
    out_path = out_dir / output_name
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def generate_for_querier(model_name, num_queriers, sizes):
    out_dir = MODEL_CONFIGS[model_name]['output_dir'] / f'querier_{num_queriers}'
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, output_name, y_label in METRIC_SPECS:
        plot_metric(model_name, metric, output_name, y_label, num_queriers, out_dir, sizes)


def main():
    parser = argparse.ArgumentParser(description='Generate computation comparison charts from data_new results.')
    parser.add_argument(
        '--models',
        nargs='*',
        choices=sorted(MODEL_CONFIGS.keys()),
        default=['decision_tree'],
        help='Model names to generate.',
    )
    parser.add_argument('--queriers', type=int, nargs='*', default=[1], help='Querier counts to generate.')
    parser.add_argument('--sizes', type=int, nargs='*', default=DEFAULT_SIZES, help='Data sizes to plot.')
    args = parser.parse_args()

    for model_name in args.models:
        for num_queriers in args.queriers:
            generate_for_querier(model_name, num_queriers, args.sizes)


if __name__ == '__main__':
    main()