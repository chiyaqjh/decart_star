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
OUT_FIXED_N_BAR = PIC_DIR / 'scalability_fixed_n_time_comm_bar.png'
OUT_FIXED_CAP_N_BAR = PIC_DIR / 'scalability_fixedN_time_comm_bar.png'


def _out_with_suffix(base_path, suffix):
    return base_path.with_name(f'{base_path.stem}_{suffix}{base_path.suffix}')

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


def _parse_pairs_file(pairs_file):
    pairs = []
    with open(pairs_file, 'r', encoding='utf-8') as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            line = line.replace(',', ' ')
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f'Invalid pair format at line {line_no}: {raw.rstrip()}')
            n_records, r_dim = int(parts[0]), int(parts[1])
            if n_records <= 0 or r_dim <= 0:
                raise ValueError(f'Pair values must be positive at line {line_no}: {raw.rstrip()}')
            pairs.append((n_records, r_dim))
    return pairs


def _parse_custom_pairs(flat_pairs, pairs_file):
    pairs = []
    if flat_pairs:
        if len(flat_pairs) % 2 != 0:
            raise ValueError('N-n-pairs must contain an even number of values (N1, n1, N2, n2, ...)')
        pairs.extend((flat_pairs[i], flat_pairs[i + 1]) for i in range(0, len(flat_pairs), 2))
    if pairs_file:
        pairs.extend(_parse_pairs_file(pairs_file))

    deduped = []
    seen = set()
    for n_records, r_dim in pairs:
        key = (int(n_records), int(r_dim))
        if key[0] <= 0 or key[1] <= 0:
            raise ValueError(f'Pair values must be positive: {key}')
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


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
    metrics = _aggregate_metrics_detailed(data)
    return metrics['lat'], metrics['comm']


def _aggregate_metrics_detailed(data):
    # Average over all model types to get one line per scheme.
    model_dict = data.get('models', {})
    enc_ms = []
    qry_ms = []
    dec_ms = []
    total_latency_ms = []
    comm_kb = []

    for mk in model_dict:
        m = model_dict[mk]
        e = np.mean(np.array(m.get('encrypt_times', [0.0]), dtype=float)) * 1000.0
        q = np.mean(np.array(m.get('query_times', [0.0]), dtype=float)) * 1000.0
        d = np.mean(np.array(m.get('decrypt_times', [0.0]), dtype=float)) * 1000.0
        c = np.mean(np.array(m.get('communication_sizes', [0.0]), dtype=float)) / 1024.0
        enc_ms.append(e)
        qry_ms.append(q)
        dec_ms.append(d)
        total_latency_ms.append(e + q + d)
        comm_kb.append(c)

    return {
        'enc': float(np.mean(enc_ms)) if enc_ms else 0.0,
        'qry': float(np.mean(qry_ms)) if qry_ms else 0.0,
        'dec': float(np.mean(dec_ms)) if dec_ms else 0.0,
        'lat': float(np.mean(total_latency_ms)) if total_latency_ms else 0.0,
        'comm': float(np.mean(comm_kb)) if comm_kb else 0.0,
    }


def _collect_curve(points, policy_size, num_runs):
    curve = {s: {'x': [], 'enc': [], 'qry': [], 'dec': [], 'lat': [], 'comm': []} for s in SCHEMES}
    used_sources = {}
    for num_records, record_dim in points:
        used_sources[(num_records, record_dim)] = {}
        for s in SCHEMES:
            data, src = _load_match(s, num_records, record_dim, policy_size, num_runs)
            metrics = _aggregate_metrics_detailed(data)
            curve[s]['x'].append((num_records, record_dim))
            curve[s]['enc'].append(metrics['enc'])
            curve[s]['qry'].append(metrics['qry'])
            curve[s]['dec'].append(metrics['dec'])
            curve[s]['lat'].append(metrics['lat'])
            curve[s]['comm'].append(metrics['comm'])
            used_sources[(num_records, record_dim)][s] = src
    return curve, used_sources


def _plot_pareto(curve, title, out_path):
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '*']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1, 1))]

    for i, scheme in enumerate(SCHEMES):
        comm = np.array(curve[scheme]['comm'], dtype=float)
        lat = np.array(curve[scheme]['lat'], dtype=float)
        order = np.argsort(comm)
        ax.plot(
            comm[order],
            lat[order],
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.8,
            markersize=6,
            markeredgewidth=0.8,
            markeredgecolor='white',
            color=COLORS.get(scheme, '#333333'),
            label=scheme,
        )

    ax.set_xlabel('Communication (KB)')
    ax.set_ylabel('Total Latency (ms)')
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def _plot_stacked_latency(curve, title, out_path):
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    x = np.arange(len(SCHEMES), dtype=float)
    enc = np.array([float(np.mean(curve[s]['enc'])) for s in SCHEMES], dtype=float)
    qry = np.array([float(np.mean(curve[s]['qry'])) for s in SCHEMES], dtype=float)
    dec = np.array([float(np.mean(curve[s]['dec'])) for s in SCHEMES], dtype=float)

    ax.bar(x, enc, color='none', edgecolor='#3A5A98', hatch='/////', linewidth=1.3, label='Encrypt')
    ax.bar(x, qry, bottom=enc, color='none', edgecolor='#5FA8A3', hatch='.....', linewidth=1.3, label='Query')
    ax.bar(x, dec, bottom=enc + qry, color='none', edgecolor='#E07A5F', hatch='xxxxx', linewidth=1.3, label='Decrypt')

    ax.set_xticks(x)
    ax.set_xticklabels(SCHEMES, rotation=15)
    ax.set_ylabel('Latency (ms)')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False, ncol=1, loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def _plot_loglog(curve, x_values, x_label, title, out_path):
    x = np.array(x_values, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.0))
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '*']
    metrics = [('lat', 'Total Latency (ms)'), ('comm', 'Communication (KB)')]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for i, scheme in enumerate(SCHEMES):
            y = np.array(curve[scheme][metric], dtype=float)
            mask = (x > 0) & (y > 0)
            if np.count_nonzero(mask) < 2:
                continue

            xm = x[mask]
            ym = y[mask]
            ax.plot(
                xm,
                ym,
                marker=markers[i % len(markers)],
                linestyle='-',
                linewidth=1.8,
                markersize=6,
                markeredgewidth=0.8,
                markeredgecolor='white',
                color=COLORS.get(scheme, '#333333'),
                label=scheme,
            )

            coeff = np.polyfit(np.log10(xm), np.log10(ym), 1)
            slope = coeff[0]
            y_fit = (10 ** coeff[1]) * (xm ** slope)
            ax.plot(xm, y_fit, linestyle=':', linewidth=1.0, color=COLORS.get(scheme, '#333333'), alpha=0.65)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.2)

    fig.suptitle(title, y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=len(labels), frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


def _plot(curve, x_values, x_label, title, out_path, use_num_records, plot_kind='line'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.8))
    x_indices = np.arange(len(x_values), dtype=float)
    x_labels = [str(v) for v in x_values]
    schemes = list(curve.keys())
    y_labels = {
        'lat': 'Total Latency (ms)',
        'comm': 'Communication (KB)',
    }

    # Slight x-offset for line plots to reveal fully-overlapped curves.
    offset_step = 0.025
    center = (len(schemes) - 1) / 2.0
    x_offsets = {s: (i - center) * offset_step for i, s in enumerate(schemes)}
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '*']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1, 1))]
    bar_hatches = ['/////', '.....', '|||||', 'xxxxx', '+++++']
    bar_width = 0.8 / max(1, len(schemes))

    for ax, metric in zip(axes, ['lat', 'comm']):  # Explicitly plot 'lat' and 'comm'
        for i, (scheme, data) in enumerate(curve.items()):
            values = data[metric]  # Extract the specific metric values
            if plot_kind == 'bar':
                x_shift = (i - center) * bar_width
                ax.bar(
                    x_indices + x_shift,
                    values,
                    width=bar_width,
                    color='none',
                    edgecolor=COLORS.get(scheme, '#333333'),
                    hatch=bar_hatches[i % len(bar_hatches)],
                    linewidth=1.3,
                    alpha=0.95,
                    label=scheme,
                    zorder=3 + i,
                )
            else:
                ax.plot(
                    x_indices + x_offsets[scheme],
                    values,
                    marker=markers[i % len(markers)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.8,
                    markersize=6,
                    markeredgewidth=0.8,
                    markeredgecolor='white',
                    alpha=0.95,
                    label=scheme,
                    zorder=3 + i,
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_labels[metric])
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        if len(x_indices) > 0:
            if plot_kind == 'bar':
                ax.set_xlim(x_indices[0] - 0.5, x_indices[-1] + 0.5)
            else:
                ax.set_xlim(x_indices[0] - 0.35, x_indices[-1] + 0.35)
        ax.grid(axis='y', alpha=0.25)

    fig.suptitle(title, y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(labels),
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
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
    parser.add_argument('--N-n-pairs', type=int, nargs='+', default=None, help='Custom (N, n) pairs: N1 n1 N2 n2 ...')
    parser.add_argument('--pairs-file', type=str, default=None, help='Text file with custom pairs, one pair per line: "N n" or "N,n"')
    parser.add_argument('--custom-only', action='store_true', help='Only generate custom (N, n) figure and skip fixed sweeps')
    parser.add_argument('--plot-kind', choices=['line', 'bar', 'both'], default='line', help='Plot style to generate')
    parser.add_argument('--extra-plots', nargs='*', default=[], help='Additional plots: 2|pareto 3|stacked 5|loglog')
    args = parser.parse_args()

    extra_alias = {'2': 'pareto', '3': 'stacked', '5': 'loglog'}
    extras = set()
    for item in args.extra_plots:
        key = str(item).strip().lower()
        extras.add(extra_alias.get(key, key))

    fixed_n_points = [(N, args.fixed_n) for N in args.N_values]
    fixed_N_points = [(args.fixed_N, n) for n in args.n_values]

    run_missing = not args.no_run_missing

    if not args.custom_only:
        print('Ensuring data for fixed n sweep...')
        _ensure_data(fixed_n_points, args.policy_size, args.num_runs, run_missing)
        curve_fixed_n, src_a = _collect_curve(fixed_n_points, args.policy_size, args.num_runs)

        x_indices = np.arange(len(args.N_values), dtype=float)
        x_labels = [str(v) for v in args.N_values]
        bar_width = 0.2
        offset = [-bar_width, 0, bar_width]
        colors = ['#3A5A98', '#5FA8A3', '#E07A5F']
        labels = ['Encrypt', 'Query', 'Decrypt']

        for scheme in SCHEMES:
            fig, ax = plt.subplots(figsize=(8, 6))
            enc = np.array(curve_fixed_n[scheme]['enc'])
            qry = np.array(curve_fixed_n[scheme]['qry'])
            dec = np.array(curve_fixed_n[scheme]['dec'])
            ax.bar(x_indices + offset[0], enc, width=bar_width, color=colors[0], label='Encrypt', alpha=0.85)
            ax.bar(x_indices + offset[1], qry, width=bar_width, color=colors[1], label='Query', alpha=0.85)
            ax.bar(x_indices + offset[2], dec, width=bar_width, color=colors[2], label='Decrypt', alpha=0.85)
            ax.set_xlabel('Total number of users $N$')
            ax.set_ylabel('Latency (ms)')
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.25)
            ax.legend(loc='upper left', frameon=False)
            fig.suptitle(f'{scheme}: Encrypt/Query/Decrypt Latency vs Total Users $N$', y=0.99)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(PIC_DIR / f'{scheme.lower()}_eqd_latency_vs_N.png', dpi=160, bbox_inches='tight')
            plt.close()
            print('Generated:', PIC_DIR / f'{scheme.lower()}_eqd_latency_vs_N.png')
            _plot_loglog(
                curve_fixed_n,
                x_values=args.N_values,
                x_label='N (num_records)',
                title=f'Log-Log Scaling (Fixed n={args.fixed_n})',
                out_path=out,
            )
            print('Generated:', out)

            out = _out_with_suffix(OUT_FIXED_CAP_N, 'loglog')
            _plot_loglog(
                curve_fixed_N,
                x_values=args.n_values,
                x_label='n (record_dim)',
                title=f'Log-Log Scaling (Fixed N={args.fixed_N})',
                out_path=out,
            )
            print('Generated:', out)

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

    custom_pairs = _parse_custom_pairs(args.N_n_pairs, args.pairs_file)
    if custom_pairs:
        print('Ensuring data for custom (N, n) pairs...')
        _ensure_data(custom_pairs, args.policy_size, args.num_runs, run_missing)
        curve_custom, _ = _collect_curve(custom_pairs, args.policy_size, args.num_runs)
        if args.plot_kind in ('line', 'both'):
            _plot(
                curve_custom,
                x_values=[f'N={N}, n={n}' for N, n in custom_pairs],
                x_label='Custom (N, n) Pairs',
                title='Total Latency and Communication vs Custom (N, n) Pairs',
                out_path=PIC_DIR / 'custom_N_n_pairs.png',
                use_num_records=False,
                plot_kind='line',
            )
            print('Generated:', PIC_DIR / 'custom_N_n_pairs.png')

        if args.plot_kind in ('bar', 'both'):
            _plot(
                curve_custom,
                x_values=[f'N={N}, n={n}' for N, n in custom_pairs],
                x_label='Custom (N, n) Pairs',
                title='Total Latency and Communication vs Custom (N, n) Pairs (Bar)',
                out_path=PIC_DIR / 'custom_N_n_pairs_bar.png',
                use_num_records=False,
                plot_kind='bar',
            )
            print('Generated:', PIC_DIR / 'custom_N_n_pairs_bar.png')

        if 'pareto' in extras:
            out = PIC_DIR / 'custom_N_n_pairs_pareto.png'
            _plot_pareto(curve_custom, 'Pareto Frontier (Custom (N, n) Pairs)', out)
            print('Generated:', out)

        if 'stacked' in extras:
            out = PIC_DIR / 'custom_N_n_pairs_stacked_latency.png'
            _plot_stacked_latency(curve_custom, 'Latency Breakdown by Scheme (Custom (N, n) Pairs)', out)
            print('Generated:', out)


if __name__ == '__main__':
    main()
