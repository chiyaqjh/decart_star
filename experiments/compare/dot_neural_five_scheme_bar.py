"""Bar charts for dot and neural_network comparisons across five schemes."""

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

TARGET_SIZE = 10000


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
            data = load_latest_result(folder, model_key, TARGET_SIZE)
            model_block = data['models'][model_key]
            summary[model_key][scheme_label] = {
                'query_s': first_metric(model_block, 'query_times'),
                'comm_gb': first_metric(model_block, 'communication_sizes') / (1024 ** 3),
            }
    return summary


def add_bars(ax, schemes, values, ylabel, title):
    x = np.arange(len(schemes))
    bars = []
    for idx, scheme in enumerate(schemes):
        style = SCHEME_STYLE[scheme]
        bar = ax.bar(
            x[idx],
            values[idx],
            width=0.72,
            color=style['facecolor'],
            edgecolor=style['edgecolor'],
            linewidth=1.5,
            hatch=style['hatch'],
            zorder=3,
        )
        bars.append(bar[0])

    ax.set_xticks(x)
    ax.set_xticklabels(schemes, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14)
    style_axes(ax)

    ymax = max(values) if values else 0.0
    offset = ymax * 0.02 if ymax > 0 else 0.1
    for bar, value in zip(bars, values):
        label = f'{value:.1f}' if value >= 100 else f'{value:.2f}'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            label,
            ha='center',
            va='bottom',
            rotation=90,
            fontsize=9,
        )


def make_plot(summary: dict) -> Path:
    scheme_labels = [label for label, _ in SCHEMES]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Five-Scheme Comparison at size=10000 (q=1, N=10000, n=32)', fontsize=18)

    for row, (model_key, model_title) in enumerate(MODELS):
        query_values = [summary[model_key][scheme]['query_s'] for scheme in scheme_labels]
        comm_values = [summary[model_key][scheme]['comm_gb'] for scheme in scheme_labels]
        add_bars(axes[row, 0], scheme_labels, query_values, 'Query Time (s)', f'{model_title} Query Time')
        add_bars(axes[row, 1], scheme_labels, comm_values, 'Communication (GiB)', f'{model_title} Communication')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = single_output_path(OUTPUT_DIR, 'dot_neural_five_scheme_bar', 'png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def main() -> None:
    summary = collect_metrics()
    out = make_plot(summary)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()