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


def _normalize_combo(combo):
    if len(combo) == 4:
        num_records, record_dim, policy_size, num_runs = combo
        return DEFAULT_N, DEFAULT_BLOCK_SIZE, num_records, record_dim, policy_size, num_runs
    return combo


def _load_match(scheme_name, N, n, num_records, record_dim, policy_size, num_runs):
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
            return data, Path(file_path).name
    return None, None


def _run_one(scheme_name, N, n, num_records, record_dim, policy_size, num_runs):
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
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def fetch_result_bundle(N, n, num_records, record_dim, policy_size, num_runs, run_missing=True):
    key = (N, n, num_records, record_dim, policy_size, num_runs)
    if key in RESULT_CACHE:
        return RESULT_CACHE[key]

    data_map = {}
    src_map = {}
    for scheme in SCHEMES:
        data, src = _load_match(scheme, N, n, num_records, record_dim, policy_size, num_runs)
        if data is None and run_missing:
            _run_one(scheme, N, n, num_records, record_dim, policy_size, num_runs)
            data, src = _load_match(scheme, N, n, num_records, record_dim, policy_size, num_runs)
        if data is None:
            raise FileNotFoundError(
                f'Missing result for {scheme}: N={N}, n={n}, num_records={num_records}, record_dim={record_dim}, '
                f'policy_size={policy_size}, num_runs={num_runs}'
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


def _phase_comm_kb(model_block):
    if model_block.get('comm_upload_sizes') and model_block.get('comm_query_sizes') and model_block.get('comm_decrypt_sizes'):
        return {
            'upload_kb': _mean_kb(model_block.get('comm_upload_sizes')),
            'query_kb': _mean_kb(model_block.get('comm_query_sizes')),
            'decrypt_kb': _mean_kb(model_block.get('comm_decrypt_sizes')),
            'total_kb': _mean_kb(model_block.get('communication_sizes')),
        }

    total_kb = _mean_kb(model_block.get('communication_sizes'))
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


def metric_for_scheme(data, metric, model_key=None):
    models = data.get('models', {})
    blocks = [models[model_key]] if model_key else list(models.values())
    values = []
    for block in blocks:
        phase = _phase_comm_kb(block)
        values.append(phase[metric])
    return float(np.mean(values)) if values else 0.0


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
        )
        for scheme in SCHEMES:
            curve[scheme].append(metric_for_scheme(data_map[scheme], metric=metric, model_key=model_key))
    return curve


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


def chart_1_register_vs_users(run_missing):
    users = [10, 100, 500, 1000, 5000, 10000]
    first_n, second_n = REGISTER_N_VALUES
    n100 = curve_from_real_data(
        users,
        combo_fn=lambda user_count: (user_count, first_n, FIXED_DATA_SIZE, FIXED_DATA_SIZE, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='upload_kb',
        run_missing=run_missing,
    )
    n1000 = curve_from_real_data(
        users,
        combo_fn=lambda user_count: (user_count, second_n, FIXED_DATA_SIZE, FIXED_DATA_SIZE, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='upload_kb',
        run_missing=run_missing,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), sharey=True)
    plot_multi_line(
        axes[0],
        users,
        n100,
        x_label='Number of users',
        y_label='Communication cost (KB)',
        title=f'Register Communication (n = {first_n})',
        x_scale='log',
    )
    plot_multi_line(
        axes[1],
        users,
        n1000,
        x_label='Number of users',
        y_label='Communication cost (KB)',
        title=f'Register Communication (n = {second_n})',
        x_scale='log',
    )
    save_figure(fig, f'01_register_vs_users_n{first_n}_n{second_n}.png')


def chart_2_encrypt_vs_records(run_missing):
    records = [10, 100, 500, 1000, 5000, 10000]
    curve = curve_from_real_data(
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
        title='Encrypt Communication vs Data Records (policy size = 32)',
        x_scale='log',
    )
    save_figure(fig, '02_encrypt_vs_records_policy32.png')


def chart_3_check_vs_model_size(run_missing):
    models = ['MLP100', 'MLP10000', 'NN100000', 'ResNet']
    labels_to_dim = {
        'MLP100': 100,
        'MLP10000': 1000,
        'NN100000': 10000,
        'ResNet': 10000,
    }

    values_map = {scheme: [] for scheme in SCHEMES}
    for label in models:
        data_map, _ = fetch_result_bundle(
            N=DEFAULT_N,
            n=DEFAULT_BLOCK_SIZE,
            num_records=FIXED_DATA_SIZE,
            record_dim=labels_to_dim[label],
            policy_size=DEFAULT_POLICY_SIZE,
            num_runs=DEFAULT_NUM_RUNS,
            run_missing=run_missing,
        )
        for scheme in SCHEMES:
            model_key = MODEL_KEY_MAP[label]
            values_map[scheme].append(metric_for_scheme(data_map[scheme], metric='query_kb', model_key=model_key))

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    plot_grouped_bar(
        ax,
        models,
        values_map,
        x_label='Model size',
        y_label='Communication cost (KB)',
        title='Check Communication vs Model Size (5 Schemes)',
    )
    save_figure(fig, '03_check_vs_model_size.png')


def chart_4_query_vs_result_size(run_missing):
    result_sizes = [10, 100, 500, 1000, 5000, 10000]
    values_map = curve_from_real_data(
        result_sizes,
        combo_fn=lambda result_size: (result_size, result_size, DEFAULT_POLICY_SIZE, DEFAULT_NUM_RUNS),
        metric='query_kb',
        run_missing=run_missing,
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    plot_multi_line(
        ax,
        result_sizes,
        values_map,
        x_label='Size of query results',
        y_label='Communication cost (KB)',
        title='Query Communication vs Query Result Size (5 Schemes)',
        x_scale='log',
    )
    save_figure(fig, '04_query_vs_result_size.png')


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
    chart_1_register_vs_users(run_missing)
    chart_2_encrypt_vs_records(run_missing)
    chart_3_check_vs_model_size(run_missing)
    chart_4_query_vs_result_size(run_missing)
    print(f'All communication charts are saved in: {OUT_DIR}')


if __name__ == '__main__':
    main()
