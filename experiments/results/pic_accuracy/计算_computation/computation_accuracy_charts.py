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

OUT_DIR = PROJECT_ROOT / 'experiments' / 'results' / 'pic_accuracy' / '计算_computation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
DEFAULT_BLOCK_SIZE = Config.BLOCK_SIZE
DEFAULT_POLICY_SIZE = Config.EXPERIMENT_POLICY_SIZE
DEFAULT_NUM_RUNS = Config.EXPERIMENT_NUM_RUNS
FIXED_DATA_SIZE = Config.EXPERIMENT_FIXED_DATA_SIZE
REGISTER_N_VALUES = list(Config.EXPERIMENT_REGISTER_N_VALUES)
MAX_SWEEP_VALUE = 5000
TARGET_MODEL_KEY = 'decision_tree'

RESULT_CACHE = {}


def style_axes(ax):
    if ax.get_yscale() == 'linear':
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', axis='y', linestyle='--', linewidth=1, color='#888', alpha=0.6, zorder=0)
    ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.7, color='#bbb', alpha=0.4, zorder=0)


def save_figure(fig, filename):
    output = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Generated: {output}')


def title_filename(title):
    return f'{title}.png'


def plot_multi_line(ax, x_values, y_map, x_label, y_label, title, x_scale='linear'):
    for idx, scheme in enumerate(SCHEMES):
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
    ax.legend(frameon=True, edgecolor='#ccc')


def plot_grouped_bar(ax, categories, values_map, x_label, y_label, title):
    x = np.arange(len(categories), dtype=float)
    bar_width = 0.15
    center = (len(SCHEMES) - 1) / 2.0

    for idx, scheme in enumerate(SCHEMES):
        ax.bar(
            x + (idx - center) * bar_width,
            values_map[scheme],
            width=bar_width,
            label=scheme,
            color=FILL_COLORS[idx],
            edgecolor=SCHEME_COLORS[idx],
            linewidth=1.4,
            hatch=HATCHES[idx],
            zorder=3,
        )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    style_axes(ax)
    ax.legend(frameon=True, edgecolor='#ccc')


def plot_grouped_bar_log(ax, categories, values_map, x_label, y_label, title):
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
    plot_grouped_bar(ax, categories, sanitized_map, x_label, y_label, title)
    ax.set_yscale('log')
    style_axes(ax)


def _mean_ms(values):
    arr = np.array(values or [0.0], dtype=float)
    return float(np.mean(arr) * 1000.0)


def _mean_ms_or_nan(values):
    if not values:
        return float('nan')
    arr = np.array(values, dtype=float)
    return float(np.mean(arr) * 1000.0)


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


def metric_for_scheme(data, metric, model_key=None):
    if data is None:
        return float('nan')

    models = data.get('models', {})
    summary = data.get('summary', {})

    def _summary_ms(block, key):
        value = block.get(key)
        if value is None:
            return float('nan')
        return float(value) * 1000.0

    if model_key is not None:
        block = models.get(model_key, {})
        summary_block = summary.get(model_key, {})
        if metric == 'check_ms':
            value = _summary_ms(summary_block, 'avg_check_time')
            if not np.isnan(value):
                return value
            if block.get('check_times'):
                return _mean_ms(block.get('check_times'))
            return float('nan')
        if metric == 'keygen_ms':
            value = _summary_ms(summary_block, 'avg_keygen_time')
            if not np.isnan(value):
                return value
            return _mean_ms_or_nan(block.get('keygen_times'))
        if metric == 'register_ms':
            value = _summary_ms(summary_block, 'avg_register_time')
            if not np.isnan(value):
                return value
            return _mean_ms_or_nan(block.get('register_times'))
        if metric == 'total_ms':
            encrypt_ms = _summary_ms(summary_block, 'avg_encrypt_time')
            query_ms = _summary_ms(summary_block, 'avg_query_time')
            decrypt_ms = _summary_ms(summary_block, 'avg_decrypt_time')
            if np.isnan(encrypt_ms):
                encrypt_ms = _mean_ms_or_nan(block.get('encrypt_times'))
            if np.isnan(query_ms):
                query_ms = _mean_ms_or_nan(block.get('query_times'))
            if np.isnan(decrypt_ms):
                decrypt_ms = _mean_ms_or_nan(block.get('decrypt_times'))
            if np.isnan(encrypt_ms) or np.isnan(query_ms) or np.isnan(decrypt_ms):
                return float('nan')
            return encrypt_ms + query_ms + decrypt_ms
        return _mean_ms_or_nan(block.get(metric))

    values = []
    for block_name, block in models.items():
        summary_block = summary.get(block_name, {})
        if metric == 'check_ms':
            value = _summary_ms(summary_block, 'avg_check_time')
            if not np.isnan(value):
                values.append(value)
                continue
            if block.get('check_times'):
                values.append(_mean_ms(block.get('check_times')))
        elif metric == 'keygen_ms':
            value = _summary_ms(summary_block, 'avg_keygen_time')
            if np.isnan(value):
                value = _mean_ms_or_nan(block.get('keygen_times'))
            if not np.isnan(value):
                values.append(value)
        elif metric == 'register_ms':
            value = _summary_ms(summary_block, 'avg_register_time')
            if np.isnan(value):
                value = _mean_ms_or_nan(block.get('register_times'))
            if not np.isnan(value):
                values.append(value)
        elif metric == 'total_ms':
            encrypt_ms = _summary_ms(summary_block, 'avg_encrypt_time')
            query_ms = _summary_ms(summary_block, 'avg_query_time')
            decrypt_ms = _summary_ms(summary_block, 'avg_decrypt_time')
            if np.isnan(encrypt_ms):
                encrypt_ms = _mean_ms_or_nan(block.get('encrypt_times'))
            if np.isnan(query_ms):
                query_ms = _mean_ms_or_nan(block.get('query_times'))
            if np.isnan(decrypt_ms):
                decrypt_ms = _mean_ms_or_nan(block.get('decrypt_times'))
            if not (np.isnan(encrypt_ms) or np.isnan(query_ms) or np.isnan(decrypt_ms)):
                values.append(encrypt_ms + query_ms + decrypt_ms)
        else:
            value = _mean_ms_or_nan(block.get(metric))
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


def blank_curve_point(curve, x_values, target_x):
    if target_x not in x_values:
        return curve
    target_index = x_values.index(target_x)
    for scheme in SCHEMES:
        curve[scheme][target_index] = float('nan')
    return curve


def chart_1_parameter_overview():
    categories = ['Total users', 'Block size', 'Records', 'Policy size', 'Record dim', 'Runs']
    profile = [DEFAULT_N, DEFAULT_BLOCK_SIZE, FIXED_DATA_SIZE, DEFAULT_POLICY_SIZE, FIXED_DATA_SIZE, DEFAULT_NUM_RUNS]
    values_map = {scheme: profile[:] for scheme in SCHEMES}

    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    plot_grouped_bar(
        ax,
        categories,
        values_map,
        x_label='Computation dimensions',
        y_label='Scale',
        title='Computation Parameter Profile (5 Schemes)',
    )
    ax.set_yscale('log')
    ax.set_xticklabels(categories, rotation=15, ha='right', fontsize=11)
    style_axes(ax)
    save_figure(fig, '01_parameter_overview.png')


def chart_2_keygen_vs_n(run_missing):
    n_values = [10, 100, 1000, MAX_SWEEP_VALUE]
    curve = curve_from_real_data(
        n_values,
        combo_fn=lambda record_dim: (FIXED_DATA_SIZE, record_dim, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='keygen_ms',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        n_values,
        curve,
        x_label='Record dimension',
        y_label='Running time (ms)',
        title='Key Generation Time vs Record Dimension (5 Schemes)',
        x_scale='log',
    )
    save_figure(fig, '02_keygen_vs_n.png')


def chart_3_register_vs_users(run_missing):
    users = [10, 100, 1000, MAX_SWEEP_VALUE]
    first_n, second_n = REGISTER_N_VALUES
    n100 = curve_from_real_data(
        users,
        combo_fn=lambda user_count: (user_count, first_n, FIXED_DATA_SIZE, FIXED_DATA_SIZE, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='register_ms',
        run_missing=run_missing,
    )
    n1000 = curve_from_real_data(
        users,
        combo_fn=lambda user_count: (user_count, second_n, FIXED_DATA_SIZE, FIXED_DATA_SIZE, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='register_ms',
        run_missing=run_missing,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), sharey=True)
    plot_multi_line(
        axes[0],
        users,
        n100,
        x_label='Number of users',
        y_label='Running time (ms)',
        title=f'Register Time (n = {first_n})',
        x_scale='log',
    )
    plot_multi_line(
        axes[1],
        users,
        n1000,
        x_label='Number of users',
        y_label='Running time (ms)',
        title=f'Register Time (n = {second_n})',
        x_scale='log',
    )
    save_figure(fig, f'03_register_vs_users_n{first_n}_n{second_n}.png')


def chart_4_encrypt_vs_records(run_missing):
    records = [10, 100, 1000, 10000]
    chart_title = f'Encrypt Time vs Number of Data Records (policy size = {DEFAULT_POLICY_SIZE})'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (DEFAULT_N, DEFAULT_BLOCK_SIZE, record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='encrypt_times',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        records,
        curve,
        x_label='Number of data records',
        y_label='Running time (ms)',
        title='',
        x_scale='log',
    )
    save_figure(fig, title_filename(chart_title))


def chart_4_encrypt_vs_records_q_bar(run_missing):
    records = [10, 100, 1000, MAX_SWEEP_VALUE]
    policy_sizes = [10, 50, 100]
    chart_title = 'Decision Tree Encrypt Time (queriers = 10, 50, 100)'

    fig, axes = plt.subplots(1, 3, figsize=(22.0, 5.7), sharey=True)
    categories = [str(value) for value in records]
    for ax, policy_size in zip(axes, policy_sizes):
        curve = curve_from_real_data(
            records,
            combo_fn=lambda record_count, policy_size=policy_size: (DEFAULT_N, DEFAULT_BLOCK_SIZE, record_count, record_count, policy_size, DEFAULT_NUM_RUNS),
            metric='encrypt_times',
            model_key=TARGET_MODEL_KEY,
            run_missing=run_missing,
        )
        plot_grouped_bar_log(
            ax,
            categories,
            curve,
            x_label='Number of data records',
            y_label='Running time (ms)',
            title='',
        )
    save_figure(fig, title_filename(chart_title))


def chart_4_encrypt_vs_records_bar(run_missing):
    records = [10, 100, 1000, 10000]
    categories = [str(value) for value in records]
    chart_title = f'Encrypt Time vs Number of Data Records (policy size = {DEFAULT_POLICY_SIZE}) bar'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (DEFAULT_N, DEFAULT_BLOCK_SIZE, record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='encrypt_times',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_grouped_bar_log(
        ax,
        categories,
        curve,
        x_label='Number of data records',
        y_label='Running time (ms)',
        title='',
    )
    save_figure(fig, title_filename(chart_title))


def chart_5_check_vs_model_size(run_missing):
    records = [10, 100, 500, 1000, 5000, 10000]
    chart_title = 'Decision Tree Check Time vs Number of Data Records (5 Schemes)'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='check_ms',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        records,
        curve,
        x_label='Number of data records',
        y_label='Running time (ms)',
        title='',
        x_scale='log',
    )
    save_figure(fig, title_filename(chart_title))


def chart_5_check_vs_model_size_bar(run_missing):
    records = [10, 100, 500, 1000, 5000, 10000]
    categories = [str(value) for value in records]
    chart_title = 'Decision Tree Check Time vs Number of Data Records (5 Schemes) bar'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='check_ms',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_grouped_bar_log(
        ax,
        categories,
        curve,
        x_label='Number of data records',
        y_label='Running time (ms)',
        title='',
    )
    save_figure(fig, title_filename(chart_title))


def chart_6_query_vs_records(run_missing):
    records = [10, 100, 500, 1000, 5000, 10000]
    chart_title = 'Decision Tree Query Time vs Number of Data Records (5 Schemes)'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='query_times',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        records,
        curve,
        x_label='Number of data records',
        y_label='Running time (ms)',
        title='',
        x_scale='log',
    )
    save_figure(fig, title_filename(chart_title))


def chart_6_query_vs_records_bar(run_missing):
    records = [10, 100, 500, 1000, 5000, 10000]
    categories = [str(value) for value in records]
    chart_title = 'Decision Tree Query Time vs Number of Data Records (5 Schemes) bar'
    curve = curve_decision_tree_only(
        records,
        combo_fn=lambda record_count: (record_count, record_count, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='query_times',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_grouped_bar_log(
        ax,
        categories,
        curve,
        x_label='Number of data records',
        y_label='Running time (ms)',
        title='',
    )
    save_figure(fig, title_filename(chart_title))


def chart_7_decrypt_vs_result_size(run_missing):
    result_sizes = [10, 100, 500, 1000, 5000, 10000]
    chart_title = 'Decision Tree Decrypt Time vs Data Scale (5 Schemes)'
    curve = curve_decision_tree_only(
        result_sizes,
        combo_fn=lambda result_size: (result_size, result_size, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='decrypt_times',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        result_sizes,
        curve,
        x_label='Number of data records / record dimension',
        y_label='Running time (ms)',
        title='',
        x_scale='log',
    )
    save_figure(fig, title_filename(chart_title))


def chart_7_decrypt_vs_result_size_bar(run_missing):
    result_sizes = [10, 100, 500, 1000, 5000, 10000]
    categories = [str(value) for value in result_sizes]
    chart_title = 'Decision Tree Decrypt Time vs Data Scale (5 Schemes) bar'
    curve = curve_decision_tree_only(
        result_sizes,
        combo_fn=lambda result_size: (result_size, result_size, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='decrypt_times',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_grouped_bar_log(
        ax,
        categories,
        curve,
        x_label='Number of data records / record dimension',
        y_label='Running time (ms)',
        title='',
    )
    save_figure(fig, title_filename(chart_title))


def main():
    global DEFAULT_N, DEFAULT_BLOCK_SIZE, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS, FIXED_DATA_SIZE, REGISTER_N_VALUES

    parser = argparse.ArgumentParser(description='Generate computation charts from real experiment JSON data.')
    parser.add_argument('--N', type=int, default=DEFAULT_N, help='Default N used to filter or run missing points.')
    parser.add_argument('--n', type=int, default=DEFAULT_BLOCK_SIZE, help='Default n used to filter or run missing points.')
    parser.add_argument('--policy-size', type=int, default=DEFAULT_POLICY_SIZE, help='Default policy size used to filter or run missing points.')
    parser.add_argument('--num-runs', type=int, default=DEFAULT_NUM_RUNS, help='Default run count used to filter or run missing points.')
    parser.add_argument('--fixed-data-size', type=int, default=FIXED_DATA_SIZE, help='Fixed data size used in non-sweep charts.')
    parser.add_argument('--register-n-values', type=int, nargs=2, metavar=('N1', 'N2'), default=REGISTER_N_VALUES, help='Two n values used for the register comparison chart.')
    parser.add_argument('--run-missing', action='store_true', help='Automatically run missing points before plotting.')
    args = parser.parse_args()

    DEFAULT_N = args.N
    DEFAULT_BLOCK_SIZE = args.n
    DEFAULT_POLICY_SIZE = args.policy_size
    DEFAULT_NUM_RUNS = args.num_runs
    FIXED_DATA_SIZE = args.fixed_data_size
    REGISTER_N_VALUES = list(args.register_n_values)
    RESULT_CACHE.clear()

    run_missing = args.run_missing

    chart_1_parameter_overview()
    chart_2_keygen_vs_n(run_missing)
    chart_3_register_vs_users(run_missing)
    chart_4_encrypt_vs_records(run_missing)
    chart_4_encrypt_vs_records_q_bar(run_missing)
    chart_4_encrypt_vs_records_bar(run_missing)
    chart_5_check_vs_model_size(run_missing)
    chart_5_check_vs_model_size_bar(run_missing)
    chart_6_query_vs_records(run_missing)
    chart_6_query_vs_records_bar(run_missing)
    chart_7_decrypt_vs_result_size(run_missing)
    chart_7_decrypt_vs_result_size_bar(run_missing)
    print(f'All computation charts are saved in: {OUT_DIR}')


if __name__ == '__main__':
    main()
