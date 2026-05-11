"""
Generate exactly one figure for paper items 1, 5, 6.

Item 1: Correctness-Cost tradeoff (success rate vs latency).
Item 5: Multi-model composite ranking.
Item 6: Protocol overview (architecture + sequence) in one figure.

Each run deletes older versions and keeps exactly one file per item.
"""

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from accuracy_style import apply_accuracy_style, get_pic_accuracy_dir

apply_accuracy_style()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / 'experiments' / 'results'
PIC_DIR = get_pic_accuracy_dir(PROJECT_ROOT)

SCHEMES = ['DeCart', 'DeCart*', 'CCS23', 'Server', 'Offline']
FOLDERS = {
    'DeCart': 'our_decart',
    'DeCart*': 'our_decart_star',
    'CCS23': 'scheme1_ccs23',
    'Server': 'scheme2_server',
    'Offline': 'scheme3_offline',
}
MODELS = [('dot', 'Dot Product'), ('decision_tree', 'Decision Tree'), ('neural_network', 'Neural Network')]
COLORS = {
    'DeCart': '#1f77b4',
    'DeCart*': '#ff7f0e',
    'CCS23': '#2ca02c',
    'Server': '#d62728',
    'Offline': '#9467bd',
}

OUT_FIG1 = PIC_DIR / 'fig1_correctness_cost_tradeoff.png'
OUT_FIG5 = PIC_DIR / 'fig5_multimodel_ranking.png'
OUT_FIG6 = PIC_DIR / 'fig6_protocol_overview.png'


def cleanup_old_outputs():
    patterns = [
        'fig1_correctness_cost_tradeoff*.png',
        'fig5_multimodel_ranking*.png',
        'fig6_protocol_overview*.png',
    ]
    for pattern in patterns:
        for p in PIC_DIR.glob(pattern):
            p.unlink(missing_ok=True)


def find_matched(folder: str, num_records: int, record_dim: int, policy_size: int, num_runs: int):
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
            return p, data
    raise RuntimeError(f'No matched file for {folder}')


def load_all(num_records=128, record_dim=128, policy_size=10, num_runs=3):
    loaded = {}
    src = {}
    for s in SCHEMES:
        p, d = find_matched(FOLDERS[s], num_records, record_dim, policy_size, num_runs)
        loaded[s] = d
        src[s] = Path(p).name
    return loaded, src


def _mean_ms(values):
    return float(np.mean(np.array(values, dtype=float)) * 1000.0)


def _mean_kb(values):
    return float(np.mean(np.array(values, dtype=float)) / 1024.0)


def _success_rate_pct(runs):
    if not runs:
        return 0.0
    ok = sum(1 for r in runs if bool(r.get('success', False)))
    return 100.0 * ok / len(runs)


def plot_item1_tradeoff(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    fig.suptitle('Item 1: Correctness-Cost Tradeoff (Success Rate vs Total Latency)', fontsize=13, fontweight='bold')

    for ax, (mk, mlabel) in zip(axes, MODELS):
        for s in SCHEMES:
            model = data[s]['models'][mk]
            lat = _mean_ms(model.get('encrypt_times', [0])) + _mean_ms(model.get('query_times', [0])) + _mean_ms(model.get('decrypt_times', [0]))
            succ = _success_rate_pct(model.get('runs', []))
            ax.scatter(lat, succ, s=80, color=COLORS[s], label=s)
            ax.annotate(s, (lat, succ), xytext=(4, 4), textcoords='offset points', fontsize=8)

        ax.set_title(mlabel)
        ax.set_xlabel('Total Latency (ms)')
        ax.grid(alpha=0.25)

    axes[0].set_ylabel('Success Rate (%)')

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.savefig(OUT_FIG1, dpi=160, bbox_inches='tight')
    plt.close()


def _normalize_lower_better(arr):
    arr = np.array(arr, dtype=float)
    amax, amin = float(np.max(arr)), float(np.min(arr))
    if abs(amax - amin) < 1e-12:
        return np.ones_like(arr)
    return (amax - arr) / (amax - amin)


def plot_item5_ranking(data):
    lat = []
    comm = []
    stab = []

    for s in SCHEMES:
        lat_per_model = []
        comm_per_model = []
        std_per_model = []
        for mk, _ in MODELS:
            model = data[s]['models'][mk]
            e = np.array(model.get('encrypt_times', [0]), dtype=float) * 1000.0
            q = np.array(model.get('query_times', [0]), dtype=float) * 1000.0
            d = np.array(model.get('decrypt_times', [0]), dtype=float) * 1000.0
            total_mean = float(np.mean(e) + np.mean(q) + np.mean(d))
            total_std = float(np.sqrt(np.std(e) ** 2 + np.std(q) ** 2 + np.std(d) ** 2))
            lat_per_model.append(total_mean)
            std_per_model.append(total_std)
            comm_per_model.append(_mean_kb(model.get('communication_sizes', [0])))

        lat.append(float(np.mean(lat_per_model)))
        comm.append(float(np.mean(comm_per_model)))
        stab.append(float(np.mean(std_per_model)))

    s_lat = _normalize_lower_better(lat)
    s_comm = _normalize_lower_better(comm)
    s_stab = _normalize_lower_better(stab)

    score = 100.0 * (0.5 * s_lat + 0.3 * s_comm + 0.2 * s_stab)

    order = np.argsort(score)[::-1]
    sorted_schemes = [SCHEMES[i] for i in order]
    sorted_score = [score[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(sorted_schemes, sorted_score, color=[COLORS[s] for s in sorted_schemes])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel('Composite Score (0-100, higher is better)')
    ax.set_title('Item 5: Multi-model Composite Ranking\n(Encrypt+Query+Decrypt latency, communication, stability)')
    ax.grid(axis='x', alpha=0.25)

    for b, v in zip(bars, sorted_score):
        ax.text(v + 1.0, b.get_y() + b.get_height() / 2, f'{v:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_FIG5, dpi=160, bbox_inches='tight')
    plt.close()


def _box(ax, xy, text, w=0.22, h=0.12, fc='#f5f5f5'):
    x, y = xy
    rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor='#333333', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9)
    return (x, y, w, h)


def _arrow(ax, src, dst):
    sx, sy = src
    dx, dy = dst
    ax.annotate('', xy=(dx, dy), xytext=(sx, sy), arrowprops=dict(arrowstyle='->', lw=1.2, color='#333333'))


def plot_item6_protocol_overview():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle('Item 6: Protocol Overview (Architecture + Sequence)', fontsize=13, fontweight='bold')

    # Left: architecture
    ax1.set_title('System Architecture')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    b_owner = _box(ax1, (0.08, 0.68), 'Data Owner', fc='#dceeff')
    b_server = _box(ax1, (0.40, 0.68), 'DB Server', fc='#ffe7cc')
    b_user = _box(ax1, (0.72, 0.68), 'Data Querier', fc='#e7f7de')
    b_curator = _box(ax1, (0.40, 0.38), 'Key Curator', fc='#f4e3ff')

    _arrow(ax1, (b_owner[0] + b_owner[2], b_owner[1] + b_owner[3] / 2), (b_server[0], b_server[1] + b_server[3] / 2))
    _arrow(ax1, (b_user[0], b_user[1] + b_user[3] / 2), (b_server[0] + b_server[2], b_server[1] + b_server[3] / 2))
    _arrow(ax1, (b_curator[0] + b_curator[2] / 2, b_curator[1] + b_curator[3]), (b_server[0] + b_server[2] / 2, b_server[1]))
    _arrow(ax1, (b_curator[0] + b_curator[2], b_curator[1] + b_curator[3] / 2), (b_user[0], b_user[1]))

    ax1.text(0.26, 0.80, 'Encrypted Data/Models', fontsize=8)
    ax1.text(0.56, 0.80, 'Encrypted Query/Response', fontsize=8)
    ax1.text(0.50, 0.52, 'Policy/Keys', fontsize=8, ha='center')

    # Right: sequence
    ax2.set_title('Protocol Sequence')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    lanes = {'Owner': 0.12, 'Curator': 0.38, 'Server': 0.62, 'Querier': 0.86}
    for name, x in lanes.items():
        ax2.plot([x, x], [0.08, 0.92], linestyle='--', color='#888888', linewidth=1)
        ax2.text(x, 0.95, name, ha='center', va='bottom', fontsize=9)

    events = [
        (0.86, '1) Setup & Key Distribution', 'Owner', 'Curator'),
        (0.72, '2) Encrypt and Upload Data/Model', 'Owner', 'Server'),
        (0.58, '3) Access Policy Publish', 'Curator', 'Server'),
        (0.44, '4) Encrypted Query', 'Querier', 'Server'),
        (0.30, '5) Encrypted Inference Response', 'Server', 'Querier'),
        (0.16, '6) Local Decrypt/Result', 'Querier', 'Querier'),
    ]

    for y, label, src, dst in events:
        xs, xd = lanes[src], lanes[dst]
        if src == dst:
            ax2.text(xs + 0.02, y, label, fontsize=8, va='center')
        else:
            _arrow(ax2, (xs, y), (xd, y))
            ax2.text((xs + xd) / 2, y + 0.02, label, fontsize=8, ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT_FIG6, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    cleanup_old_outputs()
    data, src = load_all()

    plot_item1_tradeoff(data)
    plot_item5_ranking(data)
    plot_item6_protocol_overview()

    print('Matched sources:')
    for s in SCHEMES:
        print(f'  {s}: {src[s]}')
    print(f'Generated: {OUT_FIG1}')
    print(f'Generated: {OUT_FIG5}')
    print(f'Generated: {OUT_FIG6}')


if __name__ == '__main__':
    main()
