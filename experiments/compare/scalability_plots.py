from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from accuracy_style import SCHEME_STYLES, apply_accuracy_style, style_axes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / 'experiments' / 'results' / 'data_new'
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'results' / 'pic_new' / 'scalability'

SCHEMES = {
    'DeCart': 'our_decart',
    'DeCart*': 'our_decart_star',
}

MODELS = {
    'dot': 'dot_N10000_n32',
    'neural_network': 'neural_network_N10000_n32',
}

def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def collect_model_series(model_name: str) -> dict[str, dict[str, list[float]]]:
    folder_name = MODELS[model_name]
    series: dict[str, dict[str, list[float]]] = {}

    for scheme_label, scheme_folder in SCHEMES.items():
        result_dir = RESULTS_ROOT / scheme_folder / folder_name
        points = []

        for path in sorted(result_dir.glob('*.json')):
            data = load_json(path)
            config = data.get('config', {})
            num_records = int(config.get('num_records', 0))
            record_dim = int(config.get('record_dim', 0))
            if num_records <= 0 or record_dim <= 0:
                continue

            model_data = data.get('models', {}).get(model_name, {})
            encrypt_times = np.array(model_data.get('encrypt_times', [0.0]), dtype=float)
            query_times = np.array(model_data.get('query_times', [0.0]), dtype=float)
            decrypt_times = np.array(model_data.get('decrypt_times', [0.0]), dtype=float)
            communication_sizes = np.array(model_data.get('communication_sizes', [0.0]), dtype=float)

            points.append({
                'scale': num_records,
                'record_dim': record_dim,
                'latency_ms': float((encrypt_times.mean() + query_times.mean() + decrypt_times.mean()) * 1000.0),
                'communication_mb': float(communication_sizes.mean() / (1024.0 * 1024.0)),
            })

        points.sort(key=lambda item: item['scale'])
        series[scheme_label] = {
            'x': [item['scale'] for item in points],
            'latency_ms': [item['latency_ms'] for item in points],
            'communication_mb': [item['communication_mb'] for item in points],
        }

    return series


def plot_grouped_bars(ax, series: dict[str, dict[str, list[float]]], metric: str, ylabel: str) -> None:
    all_scales = sorted({scale for values in series.values() for scale in values['x']})
    x_positions = np.arange(len(all_scales), dtype=float)
    scheme_labels = list(series.keys())
    width = 0.36

    for index, scheme_label in enumerate(scheme_labels):
        values = series[scheme_label]
        value_map = {scale: metric_value for scale, metric_value in zip(values['x'], values[metric])}
        y_values = [value_map.get(scale, np.nan) for scale in all_scales]
        offset = -width / 2 if index == 0 else width / 2
        scheme_style = SCHEME_STYLES[scheme_label]
        ax.bar(
            x_positions + offset,
            y_values,
            width=width,
            color=scheme_style['facecolor'],
            edgecolor=scheme_style['edgecolor'],
            linewidth=1.4,
            hatch=scheme_style['hatch'],
            label=scheme_label,
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(scale) for scale in all_scales])
    ax.set_xlabel('Data scale')
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.margins(x=0.04)
    style_axes(ax, grid_axis='y')
    ax.legend(frameon=False, loc='upper left')


def build_single_metric_figure(model_name: str, metric: str, ylabel: str, output_name: str) -> Path:
    series = collect_model_series(model_name)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plot_grouped_bars(ax, series, metric, ylabel)
    fig.tight_layout()
    out_path = OUTPUT_DIR / output_name
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main() -> None:
    apply_accuracy_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        build_single_metric_figure('dot', 'latency_ms', 'Total latency (ms)', 'dot_latency.png'),
        build_single_metric_figure('dot', 'communication_mb', 'Communication (MB)', 'dot_communication.png'),
        build_single_metric_figure('neural_network', 'latency_ms', 'Total latency (ms)', 'neural_network_latency.png'),
        build_single_metric_figure('neural_network', 'communication_mb', 'Communication (MB)', 'neural_network_communication.png'),
    ]

    for output in outputs:
        print(f'Generated: {output}')


if __name__ == '__main__':
    main()