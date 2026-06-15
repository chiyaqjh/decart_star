"""Grouped bar charts for decision tree comparisons at q=1 and q=100."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from accuracy_style import apply_accuracy_style, get_pic_accuracy_dir, single_output_path, style_axes


apply_accuracy_style()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / 'experiments' / 'results' / 'data_new'
OUTPUT_DIR = get_pic_accuracy_dir(str(PROJECT_ROOT))

SCHEMES = [
    ('DeCart', 'our_decart'),
    ('DeCart*', 'our_decart_star'),
    ('Naive CCS-2023', 'naive_ccs23'),
    ('Server', 'scheme2_server'),
    ('Offline', 'scheme3_offline'),
]

SCHEME_STYLE = {
    'DeCart': {'edgecolor': '#339AF0', 'facecolor': '#A9D6FF', 'hatch': '////'},
    'DeCart*': {'edgecolor': '#FF9800', 'facecolor': '#FFE0B2', 'hatch': '....'},
    'Naive CCS-2023': {'edgecolor': '#2E7D32', 'facecolor': '#C8E6C9', 'hatch': 'xxxx'},
    'Server': {'edgecolor': '#8E24AA', 'facecolor': '#E1BEE7', 'hatch': '++'},
    'Offline': {'edgecolor': '#C62828', 'facecolor': '#FFCDD2', 'hatch': '\\\\'},
}

Q_LEVELS = [1, 100]


def resolve_result_dir(folder: str, q_level: int) -> Path:
    if q_level == 100 and folder in {'our_decart', 'our_decart_star'}:
        return RESULT_ROOT / folder / 'decision_tree' / 'q=100'
    return RESULT_ROOT / folder


def first_metric(model_block: dict, key: str) -> float:
    value = model_block.get(key)
    if isinstance(value, list):
        return float(value[0]) if value else 0.0
    return float(value or 0.0)


def load_scheme_results(folder: str, q_level: int) -> dict:
    root = resolve_result_dir(folder, q_level)
    result = {}
    for path in root.glob('*.json'):
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        config = data.get('config', {})
        model_types = config.get('model_types') or list((data.get('models') or {}).keys())
        if len(model_types) != 1 or model_types[0] != 'decision_tree':
            continue
        if config.get('num_queriers') != q_level:
            continue
        size = config.get('num_records')
        model_block = data['models']['decision_tree']
        existing = result.get(size)
        candidate = {
            'file': path.name,
            'query_s': first_metric(model_block, 'query_times'),
            'comm_gb': first_metric(model_block, 'communication_sizes') / (1024 ** 3),
        }
        if existing is None or candidate['file'] > existing['file']:
            result[size] = candidate
    return result


def collect_metrics():
    summary = {}
    common_sizes = {}
    for q_level in Q_LEVELS:
        summary[q_level] = {}
        size_sets = []
        for scheme_label, folder in SCHEMES:
            scheme_results = load_scheme_results(folder, q_level)
            summary[q_level][scheme_label] = scheme_results
            size_sets.append(set(scheme_results.keys()))
        common_sizes[q_level] = sorted(set.intersection(*size_sets)) if size_sets else []
    return summary, common_sizes


def add_grouped_bars(ax, sizes, scheme_labels, value_map, ylabel, title):
    x = np.arange(len(sizes))
    width = 0.15
    for idx, scheme in enumerate(scheme_labels):
        style = SCHEME_STYLE[scheme]
        values = [value_map[scheme][size] for size in sizes]
        ax.bar(
            x + (idx - 2) * width,
            values,
            width=width,
            label=scheme,
            color=style['facecolor'],
            edgecolor=style['edgecolor'],
            linewidth=1.4,
            hatch=style['hatch'],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14)
    style_axes(ax)


def make_plot(summary, common_sizes) -> Path:
    scheme_labels = [label for label, _ in SCHEMES]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Decision Tree Comparison Across Sizes (N=10000, n=32)', fontsize=18)

    for row, q_level in enumerate(Q_LEVELS):
        sizes = common_sizes[q_level]
        query_map = {
            scheme: {size: summary[q_level][scheme][size]['query_s'] for size in sizes}
            for scheme in scheme_labels
        }
        comm_map = {
            scheme: {size: summary[q_level][scheme][size]['comm_gb'] for size in sizes}
            for scheme in scheme_labels
        }
        add_grouped_bars(axes[row, 0], sizes, scheme_labels, query_map, 'Query Time (s)', f'Decision Tree Query Time (q={q_level})')
        add_grouped_bars(axes[row, 1], sizes, scheme_labels, comm_map, 'Communication (GiB)', f'Decision Tree Communication (q={q_level})')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=True, bbox_to_anchor=(0.5, 0.995))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = single_output_path(OUTPUT_DIR, 'decision_tree_five_scheme_grouped_q', 'png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def main() -> None:
    summary, common_sizes = collect_metrics()
    print(f'Common sizes q=1: {common_sizes[1]}')
    print(f'Common sizes q=100: {common_sizes[100]}')
    out = make_plot(summary, common_sizes)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()