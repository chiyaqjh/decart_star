"""Grouped bar charts across all sizes for dot and neural network comparisons."""

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

MODELS = [
    ('dot', 'Dot Product'),
    ('neural_network', 'Neural Network'),
]

SIZES = [10, 100, 500, 1000, 5000, 10000]


def load_latest_result(folder: str, model_key: str, size: int) -> dict:
    model_dir = RESULT_ROOT / folder / f'{model_key}_N10000_n32'
    candidates = []
    for path in model_dir.glob('*.json'):
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        config = data.get('config', {})
        model_types = config.get('model_types') or list((data.get('models') or {}).keys())
        if len(model_types) != 1 or model_types[0] != model_key:
            continue
        if config.get('num_records') != size:
            continue
        candidates.append((path.name, data))

    if not candidates:
        raise FileNotFoundError(f'No {model_key} result for {folder} at size={size}')

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def first_metric(model_block: dict, key: str) -> float:
    value = model_block.get(key)
    if isinstance(value, list):
        return float(value[0]) if value else 0.0
    return float(value or 0.0)


def collect_metrics() -> dict:
    summary = {}
    for model_key, _ in MODELS:
        summary[model_key] = {}
        for scheme_label, folder in SCHEMES:
            scheme_rows = {}
            for size in SIZES:
                data = load_latest_result(folder, model_key, size)
                model_block = data['models'][model_key]
                scheme_rows[size] = {
                    'query_s': first_metric(model_block, 'query_times'),
                    'comm_gb': first_metric(model_block, 'communication_sizes') / (1024 ** 3),
                }
            summary[model_key][scheme_label] = scheme_rows
    return summary


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


def make_plot(summary: dict) -> Path:
    scheme_labels = [label for label, _ in SCHEMES]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Five-Scheme Grouped Bars Across Sizes (N=10000, n=32, q=1)', fontsize=18)

    for row, (model_key, model_title) in enumerate(MODELS):
        query_map = {
            scheme: {size: summary[model_key][scheme][size]['query_s'] for size in SIZES}
            for scheme in scheme_labels
        }
        comm_map = {
            scheme: {size: summary[model_key][scheme][size]['comm_gb'] for size in SIZES}
            for scheme in scheme_labels
        }
        add_grouped_bars(axes[row, 0], SIZES, scheme_labels, query_map, 'Query Time (s)', f'{model_title} Query Time')
        add_grouped_bars(axes[row, 1], SIZES, scheme_labels, comm_map, 'Communication (GiB)', f'{model_title} Communication')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=True, bbox_to_anchor=(0.5, 0.995))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = single_output_path(OUTPUT_DIR, 'dot_neural_five_scheme_grouped_sizes', 'png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def main() -> None:
    summary = collect_metrics()
    out = make_plot(summary)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()