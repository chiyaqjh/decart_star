"""
Generate scalability curves for:
1) Fixed n, increasing N: total latency and communication.
2) Fixed N, increasing n: total latency and communication.

If required results are missing, this script can run experiments automatically.
"""

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

from accuracy_style import apply_accuracy_style, get_pic_accuracy_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / 'experiments' / 'results'
PIC_DIR = get_pic_accuracy_dir(PROJECT_ROOT)

SCHEMES = ['DeCart', 'DeCart*', 'CCS23', 'Server', 'Offline']
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

COLORS = {
    'DeCart': '#1f77b4',
    'DeCart*': '#ff7f0e',
    'CCS23': '#2ca02c',
    'Server': '#d62728',
    'Offline': '#9467bd',
}

OUT_FIXED_N = PIC_DIR / 'scalability_fixed_n_time_comm.png'
OUT_FIXED_CAP_N = PIC_DIR / 'scalability_fixedN_time_comm.png'

PLOT_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
}

apply_accuracy_style()


def _load_match(scheme_name, num_records, record_dim, policy_size, num_runs):
    folder = SCHEME_INFO[scheme_name]['result_folder']
    files = sorted(glob.glob(str(RESULTS_ROOT / folder / '*.json')), reverse=True)
    for p in files:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg = data.get('config', {})
        if (
            cfg.get('num_records') == num_records
            and cfg.get('record_dim') == record_dim
            and cfg.get('policy_size') == policy_size
            and cfg.get('num_runs') == num_runs
        ):
            return data, Path(p).name
    return None, None


def _run_one(scheme_name, num_records, record_dim, policy_size, num_runs):
    runner = SCHEME_INFO[scheme_name]['runner']
    cmd = [
        sys.executable,
        str(runner),
        '--num-records', str(num_records),
        '--record-dim', str(record_dim),
        '--policy-size', str(policy_size),
        '--num-runs', str(num_runs),
    ]
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _ensure_data(points, policy_size, num_runs, run_missing):
    source_map = {}
    for num_records, record_dim in points:
        key = (num_records, record_dim)
        source_map[key] = {}
        for s in SCHEMES:
            data, src = _load_match(s, num_records, record_dim, policy_size, num_runs)
            if data is None and run_missing:
                _run_one(s, num_records, record_dim, policy_size, num_runs)
                data, src = _load_match(s, num_records, record_dim, policy_size, num_runs)
            if data is None:
                raise RuntimeError(
                    f'Missing result for {s} at (num_records={num_records}, record_dim={record_dim}, '
                    f'policy_size={policy_size}, num_runs={num_runs})'
                )
            source_map[key][s] = src
    return source_map


def _aggregate_metrics(data):
    # Average over all model types to get one line per scheme.
    model_dict = data.get('models', {})
    total_latency_ms = []
    comm_kb = []

    for mk in model_dict:
        m = model_dict[mk]
        e = np.mean(np.array(m.get('encrypt_times', [0.0]), dtype=float)) * 1000.0
        q = np.mean(np.array(m.get('query_times', [0.0]), dtype=float)) * 1000.0
        d = np.mean(np.array(m.get('decrypt_times', [0.0]), dtype=float)) * 1000.0
        c = np.mean(np.array(m.get('communication_sizes', [0.0]), dtype=float)) / 1024.0
        total_latency_ms.append(e + q + d)
        comm_kb.append(c)

    return float(np.mean(total_latency_ms)), float(np.mean(comm_kb))


def _collect_curve(points, policy_size, num_runs):
    curve = {s: {'x': [], 'lat': [], 'comm': []} for s in SCHEMES}
    used_sources = {}
    for num_records, record_dim in points:
        used_sources[(num_records, record_dim)] = {}
        for s in SCHEMES:
            data, src = _load_match(s, num_records, record_dim, policy_size, num_runs)
            lat, comm = _aggregate_metrics(data)
            curve[s]['x'].append((num_records, record_dim))
            curve[s]['lat'].append(lat)
            curve[s]['comm'].append(comm)
            used_sources[(num_records, record_dim)][s] = src
    return curve, used_sources


def _plot(curve, x_values, x_label, title, out_path, use_num_records):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_indices = range(len(x_values))  # Use numeric indices for x-axis

    for ax, metric in zip(axes, ['lat', 'comm']):  # Explicitly plot 'lat' and 'comm'
        for scheme, data in curve.items():
            values = data[metric]  # Extract the specific metric values
            ax.plot(x_indices, values, marker='o', label=scheme)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.legend()

    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(x_values, rotation=45, ha='right')  # Set string labels

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed-n', type=int, default=32, help='Fixed record_dim for N-sweep')
    parser.add_argument('--N-values', type=int, nargs='+', default=[16, 32, 64, 128], help='num_records values')
    parser.add_argument('--fixed-N', type=int, default=128, help='Fixed num_records for n-sweep')
    parser.add_argument('--n-values', type=int, nargs='+', default=[16, 32, 64, 128], help='record_dim values')
    parser.add_argument('--policy-size', type=int, default=8)
    parser.add_argument('--num-runs', type=int, default=2)
    parser.add_argument('--no-run-missing', action='store_true', help='Fail if required points are missing')
    parser.add_argument('--N-n-pairs', type=int, nargs='+', default=None, help='Custom (N, n) pairs for analysis')
    args = parser.parse_args()

    fixed_n_points = [(N, args.fixed_n) for N in args.N_values]
    fixed_N_points = [(args.fixed_N, n) for n in args.n_values]

    run_missing = not args.no_run_missing

    print('Ensuring data for fixed n sweep...')
    _ensure_data(fixed_n_points, args.policy_size, args.num_runs, run_missing)
    print('Ensuring data for fixed N sweep...')
    _ensure_data(fixed_N_points, args.policy_size, args.num_runs, run_missing)

    curve_fixed_n, src_a = _collect_curve(fixed_n_points, args.policy_size, args.num_runs)
    curve_fixed_N, src_b = _collect_curve(fixed_N_points, args.policy_size, args.num_runs)

    _plot(
        curve_fixed_n,
        x_values=args.N_values,
        x_label='N (num_records)',
        title=f'Fixed n={args.fixed_n}: Total Latency and Communication vs N',
        out_path=OUT_FIXED_N,
        use_num_records=True,
    )

    _plot(
        curve_fixed_N,
        x_values=args.n_values,
        x_label='n (record_dim)',
        title=f'Fixed N={args.fixed_N}: Total Latency and Communication vs n',
        out_path=OUT_FIXED_CAP_N,
        use_num_records=False,
    )

    print('Generated:', OUT_FIXED_N)
    print('Generated:', OUT_FIXED_CAP_N)

    print('\nSources used (fixed n sweep):')
    for (nrec, rdim), row in src_a.items():
        print(f'  (N={nrec}, n={rdim})')
        for s in SCHEMES:
            print(f'    {s}: {row[s]}')

    print('\nSources used (fixed N sweep):')
    for (nrec, rdim), row in src_b.items():
        print(f'  (N={nrec}, n={rdim})')
        for s in SCHEMES:
            print(f'    {s}: {row[s]}')

    if args.N_n_pairs:
        if len(args.N_n_pairs) % 2 != 0:
            raise ValueError("N-n-pairs must contain an even number of values (N1, n1, N2, n2, ...}")
        custom_pairs = [(args.N_n_pairs[i], args.N_n_pairs[i + 1]) for i in range(0, len(args.N_n_pairs), 2)]
        print('Ensuring data for custom (N, n) pairs...')
        _ensure_data(custom_pairs, args.policy_size, args.num_runs, run_missing)
        curve_custom, src_custom = _collect_curve(custom_pairs, args.policy_size, args.num_runs)
        _plot(
            curve_custom,
            x_values=[f'N={N}, n={n}' for N, n in custom_pairs],
            x_label='Custom (N, n) Pairs',
            title='Total Latency and Communication vs Custom (N, n) Pairs',
            out_path=PIC_DIR / 'custom_N_n_pairs.png',
            use_num_records=False,
        )
        print('Generated:', PIC_DIR / 'custom_N_n_pairs.png')


if __name__ == '__main__':
    main()
