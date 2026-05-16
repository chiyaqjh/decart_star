import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


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

OUT_DIR = PROJECT_ROOT / 'experiments' / 'results' / 'pic_accuracy' / '通信_communication'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SIZE_OUT_DIR = PROJECT_ROOT / 'experiments' / 'results' / 'pic_accuracy' / '尺寸'
SIZE_OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEMES = ['DeCart', 'DeCart*', 'CCS23', 'Server', 'Offline']
SCHEME_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
FILL_COLORS = ['#90CAF9', '#FFE0B2', '#C8E6C9', '#E1BEE7', '#FFCDD2']
HATCHES = ['//', 'xx', '..', 'oo', '++']
LINE_MARKERS = ['o', 's', '^', 'D', 'v']

SCHEME_INFO = {
    'DeCart': {
        'result_folder': 'our_decart',
        'runner': PROJECT_ROOT / 'experiments' / 'our_decart' / 'runner.py',
    },
    'DeCart*': {
        'result_folder': 'our_decart_star',
        'runner': PROJECT_ROOT / 'experiments' / 'our_decart_star' / 'runner.py',
    },
    'CCS23': {
        'result_folder': 'scheme1_ccs23',
        'runner': PROJECT_ROOT / 'experiments' / 'scheme1_ccs23' / 'runner.py',
    },
    'Server': {
        'result_folder': 'scheme2_server',
        'runner': PROJECT_ROOT / 'experiments' / 'scheme2_server' / 'runner.py',
    },
    'Offline': {
        'result_folder': 'scheme3_offline',
        'runner': PROJECT_ROOT / 'experiments' / 'scheme3_offline' / 'runner.py',
    },
}

MODEL_KEY_MAP = {
    'MLP100': 'dot',
    'MLP10000': 'decision_tree',
    'NN100000': 'neural_network',
    'ResNet': 'neural_network',
}

DEFAULT_N = Config.MAX_USERS
DEFAULT_BLOCK_SIZE = 128
DEFAULT_POLICY_SIZE = Config.EXPERIMENT_POLICY_SIZE
DEFAULT_NUM_RUNS = 1
FIXED_DATA_SIZE = Config.EXPERIMENT_FIXED_DATA_SIZE
REGISTER_N_VALUES = [32, 128]
TARGET_MODEL_KEY = 'decision_tree'

RESULT_CACHE = {}


def style_axes(ax):
    if ax.get_yscale() == 'linear':
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='y', linestyle='--', linewidth=1, color='#888', alpha=0.6, zorder=0)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.7, color='#bbb', alpha=0.4, zorder=0)


def save_figure(fig, filename, out_dir=None):
    output_dir = out_dir or OUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / filename
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Generated: {output}')


def title_filename(title):
    return f'{title}.png'


def plot_multi_line(ax, x_values, y_map, x_label, y_label, title, x_scale='linear', schemes=None, legend_kwargs=None):
    schemes = schemes or SCHEMES
    for scheme in schemes:
        idx = SCHEMES.index(scheme)
        ax.plot(
            x_values,
            y_map[scheme],
            color=SCHEME_COLORS[idx],
            marker=LINE_MARKERS[idx],
            linewidth=2.2,
            markersize=6.8,
            label=scheme,
            zorder=3,
        )
    if x_scale == 'log':
        ax.set_xscale('log')
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(v) for v in x_values], fontsize=12)
    style_axes(ax)
    legend_options = {'frameon': True, 'edgecolor': '#ccc'}
    if legend_kwargs:
        legend_options.update(legend_kwargs)
    ax.legend(**legend_options)


def plot_grouped_bar(ax, categories, values_map, x_label, y_label, title, schemes=None, legend_kwargs=None):
    schemes = schemes or SCHEMES
    x = np.arange(len(categories), dtype=float)
    bar_width = 0.15 if len(schemes) > 2 else 0.26
    center = (len(schemes) - 1) / 2.0

    for idx, scheme in enumerate(schemes):
        scheme_idx = SCHEMES.index(scheme)
        ax.bar(
            x + (idx - center) * bar_width,
            values_map[scheme],
            width=bar_width,
            label=scheme,
            color=FILL_COLORS[scheme_idx],
            edgecolor=SCHEME_COLORS[scheme_idx],
            linewidth=1.4,
            hatch=HATCHES[scheme_idx],
            zorder=3,
        )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    style_axes(ax)
    legend_options = {'frameon': True, 'edgecolor': '#ccc'}
    if legend_kwargs:
        legend_options.update(legend_kwargs)
    ax.legend(**legend_options)


def plot_grouped_bar_log(ax, categories, values_map, x_label, y_label, title, schemes=None, legend_kwargs=None):
    positive_values = [
        value
        for scheme_values in values_map.values()
        for value in scheme_values
        if np.isfinite(value) and value > 0
    ]
    floor_value = min(positive_values) / 5.0 if positive_values else 1e-3
    sanitized_map = {
        scheme: [value if np.isfinite(value) and value > 0 else floor_value for value in scheme_values]
        for scheme, scheme_values in values_map.items()
    }
    plot_grouped_bar(ax, categories, sanitized_map, x_label, y_label, title, schemes=schemes, legend_kwargs=legend_kwargs)
    ax.set_yscale('log')
    style_axes(ax)


def _series_identical(y_map, schemes):
    if len(schemes) < 2:
        return False
    first = np.array(y_map[schemes[0]], dtype=float)
    for scheme in schemes[1:]:
        other = np.array(y_map[scheme], dtype=float)
        if not np.allclose(first, other, equal_nan=True):
            return False
    return True


def _normalize_combo(combo):
    if len(combo) == 4:
        num_records, record_dim, policy_size, num_runs = combo
        return DEFAULT_N, DEFAULT_BLOCK_SIZE, num_records, record_dim, policy_size, num_runs
    return combo


def _load_match(scheme_name, N, n, num_records, record_dim, policy_size, num_runs, model_key=None, metric=None):
    folder = SCHEME_INFO[scheme_name]['result_folder']
    files = sorted(glob.glob(str(RESULTS_ROOT / folder / '*.json')), reverse=True)
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        cfg = data.get('config', {})
        if (
            cfg.get('N') == N
            and cfg.get('n') == n
            and cfg.get('num_records') == num_records
            and cfg.get('record_dim') == record_dim
            and cfg.get('policy_size') == policy_size
            and cfg.get('num_runs') == num_runs
        ):
            if model_key is not None:
                value = metric_for_scheme(data, metric=metric, model_key=model_key)
                if np.isnan(value):
                    continue
            return data, Path(file_path).name
    return None, None


def _run_one(scheme_name, N, n, num_records, record_dim, policy_size, num_runs, model_key=None):
    runner = SCHEME_INFO[scheme_name]['runner']
    cmd = [
        sys.executable,
        str(runner),
        '--N',
        str(N),
        '--n',
        str(n),
        '--num-records',
        str(num_records),
        '--record-dim',
        str(record_dim),
        '--policy-size',
        str(policy_size),
        '--num-runs',
        str(num_runs),
    ]
    if model_key is not None:
        cmd.extend(['--model-types', model_key])
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def fetch_result_bundle(N, n, num_records, record_dim, policy_size, num_runs, run_missing=True, model_key=None, metric=None):
    key = (N, n, num_records, record_dim, policy_size, num_runs, model_key, metric)
    if key in RESULT_CACHE:
        return RESULT_CACHE[key]

    data_map = {}
    src_map = {}
    for scheme in SCHEMES:
        data, src = _load_match(
            scheme,
            N,
            n,
            num_records,
            record_dim,
            policy_size,
            num_runs,
            model_key=model_key,
            metric=metric,
        )
        if data is None and run_missing:
            _run_one(scheme, N, n, num_records, record_dim, policy_size, num_runs, model_key=model_key)
            data, src = _load_match(
                scheme,
                N,
                n,
                num_records,
                record_dim,
                policy_size,
                num_runs,
                model_key=model_key,
                metric=metric,
            )
        if data is None:
            print(
                'SKIP:',
                f'{scheme} missing for N={N}, n={n}, num_records={num_records}, record_dim={record_dim}, '
                f'policy_size={policy_size}, num_runs={num_runs}, model_key={model_key}, metric={metric}'
            )
        data_map[scheme] = data
        src_map[scheme] = src

    RESULT_CACHE[key] = (data_map, src_map)
    return RESULT_CACHE[key]


def _mean_ms(values):
    arr = np.array(values or [0.0], dtype=float)
    return float(np.mean(arr) * 1000.0)


def _mean_kb(values):
    arr = np.array(values or [0.0], dtype=float)
    return float(np.mean(arr) / 1024.0)


def _sum_kb(values):
    arr = np.array(values or [0.0], dtype=float)
    return float(np.sum(arr) / 1024.0)


def _phase_comm_kb(model_block):
    if model_block.get('comm_upload_sizes') and model_block.get('comm_query_sizes') and model_block.get('comm_decrypt_sizes'):
        upload_kb = _mean_kb(model_block.get('comm_upload_sizes'))
        query_kb = _mean_kb(model_block.get('comm_query_sizes'))
        decrypt_kb = _mean_kb(model_block.get('comm_decrypt_sizes'))
        return {
            'upload_kb': upload_kb,
            'query_kb': query_kb,
            'decrypt_kb': decrypt_kb,
            'total_kb': upload_kb + query_kb + decrypt_kb,
        }

    total_kb = _sum_kb(model_block.get('communication_sizes'))
    enc_t = max(_mean_ms(model_block.get('encrypt_times')) / 1000.0, 0.0)
    qry_t = max(_mean_ms(model_block.get('query_times')) / 1000.0, 0.0)
    dec_t = max(_mean_ms(model_block.get('decrypt_times')) / 1000.0, 0.0)
    total_t = enc_t + qry_t + dec_t
    if total_t <= 0:
        ratios = {'upload_kb': 1.0 / 3.0, 'query_kb': 1.0 / 3.0, 'decrypt_kb': 1.0 / 3.0}
    else:
        ratios = {
            'upload_kb': enc_t / total_t,
            'query_kb': qry_t / total_t,
            'decrypt_kb': dec_t / total_t,
        }
    return {
        'upload_kb': total_kb * ratios['upload_kb'],
        'query_kb': total_kb * ratios['query_kb'],
        'decrypt_kb': total_kb * ratios['decrypt_kb'],
        'total_kb': total_kb,
    }


def _register_comm_kb(model_block):
    total_sizes = model_block.get('register_total_auxiliary_sizes')
    if total_sizes:
        return _mean_kb(total_sizes)

    component_groups = [
        model_block.get('register_crs_sizes'),
        model_block.get('register_pp_sizes'),
        model_block.get('register_aux_sizes'),
    ]
    if any(component_groups):
        values = [_mean_kb(values) for values in component_groups if values]
        return float(sum(values)) if values else float('nan')

    return float('nan')


def _structure_size_kb(model_block, metric):
    metric_map = {
        'setup_crs_kb': 'setup_crs_sizes',
        'setup_pp_kb': 'setup_pp_sizes',
        'setup_aux_kb': 'setup_aux_sizes',
        'setup_total_aux_kb': 'setup_total_auxiliary_sizes',
        'register_crs_kb': 'register_crs_sizes',
        'register_pp_kb': 'register_pp_sizes',
        'register_aux_kb': 'register_aux_sizes',
        'register_total_aux_kb': 'register_total_auxiliary_sizes',
        'final_crs_kb': 'final_crs_sizes',
        'final_pp_kb': 'final_pp_sizes',
        'final_aux_kb': 'final_aux_sizes',
        'final_total_aux_kb': 'final_total_auxiliary_sizes',
    }
    field_name = metric_map.get(metric)
    if field_name is None:
        return float('nan')
    values = model_block.get(field_name)
    return _mean_kb(values) if values else float('nan')


def metric_for_scheme(data, metric, model_key=None):
    if data is None:
        return float('nan')

    models = data.get('models', {})
    if model_key is not None:
        block = models.get(model_key)
        if not block:
            return float('nan')
        if metric == 'register_kb':
            return _register_comm_kb(block)
        if metric.endswith('_kb') and metric not in {'upload_kb', 'query_kb', 'decrypt_kb', 'total_kb'}:
            return _structure_size_kb(block, metric)
        phase = _phase_comm_kb(block)
        return phase[metric]

    blocks = list(models.values())
    values = []
    for block in blocks:
        if metric == 'register_kb':
            value = _register_comm_kb(block)
        elif metric.endswith('_kb') and metric not in {'upload_kb', 'query_kb', 'decrypt_kb', 'total_kb'}:
            value = _structure_size_kb(block, metric)
        else:
            phase = _phase_comm_kb(block)
            value = phase[metric]
        if not np.isnan(value):
            values.append(value)
    return float(np.mean(values)) if values else float('nan')


def curve_from_real_data(x_values, combo_fn, metric, model_key=None, run_missing=True):
    curve = {scheme: [] for scheme in SCHEMES}
    for x in x_values:
        N, n, num_records, record_dim, policy_size, num_runs = _normalize_combo(combo_fn(x))
        data_map, _ = fetch_result_bundle(
            N=N,
            n=n,
            num_records=num_records,
            record_dim=record_dim,
            policy_size=policy_size,
            num_runs=num_runs,
            run_missing=run_missing,
            model_key=model_key,
            metric=metric,
        )
        for scheme in SCHEMES:
            curve[scheme].append(metric_for_scheme(data_map[scheme], metric=metric, model_key=model_key))
    return curve


def curve_decision_tree_only(x_values, combo_fn, metric, run_missing=True):
    return curve_from_real_data(
        x_values,
        combo_fn=combo_fn,
        metric=metric,
        model_key=TARGET_MODEL_KEY,
        run_missing=run_missing,
    )


def latest_config_for_scheme(scheme):
    folder = RESULTS_ROOT / SCHEME_INFO[scheme]['result_folder']
    files = sorted(glob.glob(str(folder / '*.json')), reverse=True)
    if not files:
        return None
    with open(files[0], 'r', encoding='utf-8') as handle:
        return json.load(handle).get('config', {})


def chart_0_parameter_overview():
    categories = ['Total users', 'Block size', 'Records', 'Policy size', 'Record dim', 'Runs']
    profile = [DEFAULT_N, DEFAULT_BLOCK_SIZE, FIXED_DATA_SIZE, DEFAULT_POLICY_SIZE, FIXED_DATA_SIZE, DEFAULT_NUM_RUNS]
    values_map = {scheme: profile[:] for scheme in SCHEMES}

    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    plot_grouped_bar(
        ax,
        categories,
        values_map,
        x_label='Communication dimensions',
        y_label='Scale',
        title='Communication Parameter Profile (5 Schemes)',
    )
    ax.set_yscale('log')
    ax.set_xticklabels(categories, rotation=15, ha='right', fontsize=11)
    style_axes(ax)
    save_figure(fig, '00_parameter_overview.png')


def chart_1_register_vs_n(run_missing):
    n_values = [16, 32, 64, 128, 256, 512]
    decart_schemes = ['DeCart', 'DeCart*']
    chart_title = f'Register Communication vs Block Size (N = {DEFAULT_N})'
    curve = curve_decision_tree_only(
        n_values,
        combo_fn=lambda n_value: (DEFAULT_N, n_value, FIXED_DATA_SIZE, FIXED_DATA_SIZE, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='register_kb',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        n_values,
        curve,
        x_label='Block size n',
        y_label='Communication cost (KB)',
        title='',
        x_scale='log',
        schemes=decart_schemes,
    )
    save_figure(fig, title_filename(chart_title))


def chart_2_encrypt_vs_records(run_missing):
    records = [10, 100, 1000, 10000]
    chart_title = f'Encrypt Communication vs Number of Data Records (policy size = {DEFAULT_POLICY_SIZE})'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='upload_kb',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        records,
        curve,
        x_label='Number of data records',
        y_label='Communication cost (KB)',
        title='',
        x_scale='log',
    )
    save_figure(fig, title_filename(chart_title))


def chart_2_encrypt_vs_records_bar(run_missing):
    records = [10, 100, 1000, 10000]
    categories = [str(value) for value in records]
    chart_title = f'Encrypt Communication vs Number of Data Records (policy size = {DEFAULT_POLICY_SIZE}) bar'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='upload_kb',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_grouped_bar_log(
        ax,
        categories,
        curve,
        x_label='Number of data records',
        y_label='Communication cost (KB)',
        title='',
    )
    save_figure(fig, title_filename(chart_title))


def chart_3_check_vs_records(run_missing):
    records = [10, 100, 500, 1000, 5000, 10000]
    chart_title = 'Decision Tree Total Communication vs Number of Data Records (5 Schemes)'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='total_kb',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        records,
        curve,
        x_label='Number of data records',
        y_label='Communication cost (KB)',
        title='',
        x_scale='log',
    )
    save_figure(fig, title_filename(chart_title))


def chart_3_check_vs_records_bar(run_missing):
    records = [10, 100, 500, 1000, 5000, 10000]
    categories = [str(value) for value in records]
    chart_title = 'Decision Tree Total Communication vs Number of Data Records (5 Schemes) bar'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='total_kb',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_grouped_bar_log(
        ax,
        categories,
        curve,
        x_label='Number of data records',
        y_label='Communication cost (KB)',
        title='',
    )
    save_figure(fig, title_filename(chart_title))


def chart_4_query_vs_records(run_missing):
    records = [10, 100, 500, 1000, 5000, 10000]
    chart_title = 'Decision Tree Query Communication vs Number of Data Records (5 Schemes)'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='query_kb',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        records,
        curve,
        x_label='Number of data records',
        y_label='Communication cost (KB)',
        title='',
        x_scale='log',
        legend_kwargs={'loc': 'upper left', 'bbox_to_anchor': (1.01, 1.0), 'borderaxespad': 0.0},
    )
    save_figure(fig, title_filename(chart_title))


def chart_4_query_vs_records_bar(run_missing):
    records = [10, 100, 500, 1000, 5000, 10000]
    categories = [str(value) for value in records]
    chart_title = 'Decision Tree Query Communication vs Number of Data Records (5 Schemes) bar'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='query_kb',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_grouped_bar_log(
        ax,
        categories,
        curve,
        x_label='Number of data records',
        y_label='Communication cost (KB)',
        title='',
        legend_kwargs={'loc': 'upper left', 'bbox_to_anchor': (1.01, 1.0), 'borderaxespad': 0.0},
    )
    save_figure(fig, title_filename(chart_title))


def chart_5_decrypt_vs_result_size(run_missing):
    result_sizes = [10, 100, 500, 1000, 5000, 10000]
    chart_title = 'Decision Tree Decrypt Communication vs Data Scale (5 Schemes)'
    curve = curve_decision_tree_only(
        result_sizes,
        combo_fn=lambda result_size: (result_size, result_size, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='decrypt_kb',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        result_sizes,
        curve,
        x_label='Number of data records / record dimension',
        y_label='Communication cost (KB)',
        title='',
        x_scale='log',
    )
    save_figure(fig, title_filename(chart_title))


def chart_5_decrypt_vs_result_size_bar(run_missing):
    result_sizes = [10, 100, 500, 1000, 5000, 10000]
    categories = [str(value) for value in result_sizes]
    chart_title = 'Decision Tree Decrypt Communication vs Data Scale (5 Schemes) bar'
    curve = curve_decision_tree_only(
        result_sizes,
        combo_fn=lambda result_size: (result_size, result_size, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='decrypt_kb',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_grouped_bar_log(
        ax,
        categories,
        curve,
        x_label='Number of data records / record dimension',
        y_label='Communication cost (KB)',
        title='',
    )
    save_figure(fig, title_filename(chart_title))


def _plot_structure_breakdown_vs_n(metric_specs, title_prefix, output_name, run_missing):
    n_values = [16, 32, 64, 128, 256, 512]
    decart_schemes = ['DeCart', 'DeCart*']
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    axes = axes.flatten()

    for ax, (metric, panel_title) in zip(axes, metric_specs):
        curve = curve_decision_tree_only(
            n_values,
            combo_fn=lambda n_value: (DEFAULT_N, n_value, FIXED_DATA_SIZE, FIXED_DATA_SIZE, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
            metric=metric,
            run_missing=run_missing,
        )
        if metric.endswith('_pp_kb') or _series_identical(curve, decart_schemes):
            plot_grouped_bar(
                ax,
                [str(value) for value in n_values],
                curve,
                x_label='Block size n',
                y_label='Size (KB)',
                title=panel_title,
                schemes=decart_schemes,
            )
        else:
            plot_multi_line(
                ax,
                n_values,
                curve,
                x_label='Block size n',
                y_label='Size (KB)',
                title=panel_title,
                x_scale='log',
                schemes=decart_schemes,
            )

        if metric.endswith('_crs_kb') or metric.endswith('_total_aux_kb'):
            ax.set_yscale('log')
            ax.set_ylabel('Size (KB, log scale)', fontsize=14)
            style_axes(ax)

    fig.suptitle(f'{title_prefix} Size vs Block Size (N = {DEFAULT_N})', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_name, out_dir=SIZE_OUT_DIR)


def chart_6_setup_size_breakdown_vs_n(run_missing):
    metric_specs = [
        ('setup_crs_kb', 'Setup CRS Size'),
        ('setup_pp_kb', 'Setup PP Size'),
        ('setup_aux_kb', 'Setup AUX Size'),
        ('setup_total_aux_kb', 'Setup Total Auxiliary Size'),
    ]
    _plot_structure_breakdown_vs_n(metric_specs, 'Setup', 'communication_06_setup_size_breakdown_vs_n.png', run_missing)


def chart_7_register_size_breakdown_vs_n(run_missing):
    metric_specs = [
        ('register_crs_kb', 'Register CRS Size'),
        ('register_pp_kb', 'Register PP Size'),
        ('register_aux_kb', 'Register AUX Size'),
        ('register_total_aux_kb', 'Register Total Auxiliary Size'),
    ]
    _plot_structure_breakdown_vs_n(metric_specs, 'Register', 'communication_07_register_size_breakdown_vs_n.png', run_missing)


def chart_8_final_size_breakdown_vs_n(run_missing):
    metric_specs = [
        ('final_crs_kb', 'Final CRS Size'),
        ('final_pp_kb', 'Final PP Size'),
        ('final_aux_kb', 'Final AUX Size'),
        ('final_total_aux_kb', 'Final Total Auxiliary Size'),
    ]
    _plot_structure_breakdown_vs_n(metric_specs, 'Final', 'communication_08_final_size_breakdown_vs_n.png', run_missing)


def main():
    global DEFAULT_N, DEFAULT_BLOCK_SIZE, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS, FIXED_DATA_SIZE, REGISTER_N_VALUES

    parser = argparse.ArgumentParser(description='Generate communication charts from real experiment JSON data.')
    parser.add_argument('--N', type=int, default=DEFAULT_N, help='Default N used to filter or run missing points.')
    parser.add_argument('--n', type=int, default=DEFAULT_BLOCK_SIZE, help='Default n used to filter or run missing points.')
    parser.add_argument('--policy-size', type=int, default=DEFAULT_POLICY_SIZE, help='Default policy size used to filter or run missing points.')
    parser.add_argument('--num-runs', type=int, default=DEFAULT_NUM_RUNS, help='Default run count used to filter or run missing points.')
    parser.add_argument('--fixed-data-size', type=int, default=FIXED_DATA_SIZE, help='Fixed data size used in non-sweep charts.')
    parser.add_argument('--register-n-values', type=int, nargs=2, metavar=('N1', 'N2'), default=REGISTER_N_VALUES, help='Two n values used for the register comparison chart.')
    parser.add_argument('--no-run-missing', action='store_true', help='Do not run missing points automatically.')
    args = parser.parse_args()

    DEFAULT_N = args.N
    DEFAULT_BLOCK_SIZE = args.n
    DEFAULT_POLICY_SIZE = args.policy_size
    DEFAULT_NUM_RUNS = args.num_runs
    FIXED_DATA_SIZE = args.fixed_data_size
    REGISTER_N_VALUES = list(args.register_n_values)
    RESULT_CACHE.clear()

    run_missing = not args.no_run_missing

    chart_0_parameter_overview()
    chart_1_register_vs_n(run_missing)
    chart_2_encrypt_vs_records(run_missing)
    chart_2_encrypt_vs_records_bar(run_missing)
    chart_3_check_vs_records(run_missing)
    chart_3_check_vs_records_bar(run_missing)
    chart_4_query_vs_records(run_missing)
    chart_4_query_vs_records_bar(run_missing)
    chart_5_decrypt_vs_result_size(run_missing)
    chart_5_decrypt_vs_result_size_bar(run_missing)
    chart_6_setup_size_breakdown_vs_n(run_missing)
    chart_7_register_size_breakdown_vs_n(run_missing)
    chart_8_final_size_breakdown_vs_n(run_missing)
    print(f'All communication charts are saved in: {OUT_DIR}')


if __name__ == '__main__':
    main()
