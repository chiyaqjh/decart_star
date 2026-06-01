import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3]
SOURCE_SCRIPT = PROJECT_ROOT / 'experiments' / 'results' / 'pic_accuracy' / '通信_communication' / 'communication_accuracy_charts.py'


def _load_source_module():
    spec = importlib.util.spec_from_file_location('size_source_charts', SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f'Failed to load chart source script: {SOURCE_SCRIPT}')
    spec.loader.exec_module(module)
    return module


def _render_structure_plot(charts, metric, output_name, run_missing):
    n_values = [16, 32, 64, 128, 256, 512]
    decart_schemes = ['DeCart', 'DeCart*']
    curve = charts.curve_decision_tree_only(
        n_values,
        combo_fn=lambda n_value: (
            charts.DEFAULT_N,
            n_value,
            charts.FIXED_DATA_SIZE,
            charts.FIXED_DATA_SIZE,
            charts.DEFAULT_POLICY_SIZE,
            charts.DEFAULT_NUM_RUNS,
        ),
        metric=metric,
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    if metric.endswith('_pp_kb') or charts._series_identical(curve, decart_schemes):
        charts.plot_grouped_bar(
            ax,
            [str(value) for value in n_values],
            curve,
            x_label='Block size n',
            y_label='Size (KB)',
            title='',
            schemes=decart_schemes,
        )
    else:
        charts.plot_multi_line(
            ax,
            n_values,
            curve,
            x_label='Block size n',
            y_label='Size (KB)',
            title='',
            x_scale='log',
            schemes=decart_schemes,
        )

    if metric.endswith('_crs_kb') or metric.endswith('_total_aux_kb'):
        ax.set_yscale('log')
        ax.set_ylabel('Size (KB)', fontsize=14)
        charts.style_axes(ax)

    charts.save_figure(fig, output_name, out_dir=CURRENT_DIR)


def main(run_missing=False):
    charts = _load_source_module()

    plot_specs = [
        ('setup_crs_kb', 'setup_crs_size_vs_n.png'),
        ('setup_pp_kb', 'setup_pp_size_vs_n.png'),
        ('setup_aux_kb', 'setup_aux_size_vs_n.png'),
        ('setup_total_aux_kb', 'setup_total_auxiliary_size_vs_n.png'),
        ('register_crs_kb', 'register_crs_size_vs_n.png'),
        ('register_pp_kb', 'register_pp_size_vs_n.png'),
        ('register_aux_kb', 'register_aux_size_vs_n.png'),
        ('register_total_aux_kb', 'register_total_auxiliary_size_vs_n.png'),
        ('final_crs_kb', 'final_crs_size_vs_n.png'),
        ('final_pp_kb', 'final_pp_size_vs_n.png'),
        ('final_aux_kb', 'final_aux_size_vs_n.png'),
        ('final_total_aux_kb', 'final_total_auxiliary_size_vs_n.png'),
    ]

    for metric, output_name in plot_specs:
        _render_structure_plot(charts, metric, output_name, run_missing=run_missing)

    print(f'Generated 12 size charts in: {CURRENT_DIR}')


if __name__ == '__main__':
    main(run_missing=False)