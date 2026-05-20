import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from accuracy_style import apply_accuracy_style


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIC_OUT_DIR = PROJECT_ROOT / 'experiments' / 'results' / 'pic_new'
PIC_OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEMES = ['DeCart', 'DeCart*', 'Naive CCS-2023', 'Server', 'Offline', 'SecPQ']
RESULT_DIRS = {
    'DeCart': 'our_decart',
    'DeCart*': 'our_decart_star',
    'Naive CCS-2023': 'naive_ccs23',
    'Server': 'scheme2_server',
    'Offline': 'scheme3_offline',
    'SecPQ': 'secpq',
}
SIZE_VALUES = [32, 64, 128]
MODEL_KEYS = ['dot', 'decision_tree', 'neural_network']
PHASES = [
    ('encrypt_times', 'Encrypt Request Packet'),
    ('query_times', 'Query Request Packet'),
    ('decrypt_times', 'Response Packet'),
]

apply_accuracy_style()

# accuracy.py palette + hatch style
EDGE_COLORS = ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#8D6E63']
HATCHES = ['-----', '/////', '|||||', '.....', 'xxxxx', '+++++']


def find_latest_result_by_size(scheme: str, size: int, policy_size: int = None, num_runs: int = None):
    folder = PROJECT_ROOT / 'experiments' / 'results' / RESULT_DIRS[scheme]
    files = sorted(glob.glob(str(folder / '*.json')), reverse=True)
    for p in files:
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        cfg = d.get('config', {})
        if cfg.get('num_records') != size or cfg.get('record_dim') != size:
            continue
        if policy_size is not None and cfg.get('policy_size') != policy_size:
            continue
        if num_runs is not None and cfg.get('num_runs') != num_runs:
            continue
        return d

    # Fallback: same size regardless of policy/runs.
    for p in files:
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        cfg = d.get('config', {})
        if cfg.get('num_records') == size and cfg.get('record_dim') == size:
            return d

    raise RuntimeError(f'No matched result for {scheme}, size={size}')


def load_schemes_for_sizes(sizes):
    out = {}
    for size in sizes:
        out[size] = {}
        for s in SCHEMES:
            out[size][s] = find_latest_result_by_size(s, size)
    return out


def get_mean(data_dict, scheme, model_key, metric):
    arr = data_dict[scheme]['models'][model_key].get(metric, [])
    if not arr:
        return 0.0
    return float(np.mean(arr))


def compute_comm_phase_kb(size_data):
    comm_by_size = {}
    for size in SIZE_VALUES:
        comm_by_size[size] = {}
        for scheme in SCHEMES:
            phase_split_kb = {k: [] for k, _ in PHASES}
            for mk in MODEL_KEYS:
                model_block = size_data[size][scheme]['models'][mk]
                has_measured = (
                    ('comm_upload_sizes' in model_block and len(model_block['comm_upload_sizes']) > 0)
                    and ('comm_query_sizes' in model_block and len(model_block['comm_query_sizes']) > 0)
                    and ('comm_decrypt_sizes' in model_block and len(model_block['comm_decrypt_sizes']) > 0)
                )

                if has_measured:
                    phase_split_kb['encrypt_times'].append(float(np.mean(model_block['comm_upload_sizes'])) / 1024.0)
                    phase_split_kb['query_times'].append(float(np.mean(model_block['comm_query_sizes'])) / 1024.0)
                    phase_split_kb['decrypt_times'].append(float(np.mean(model_block['comm_decrypt_sizes'])) / 1024.0)
                else:
                    total_kb = get_mean(size_data[size], scheme, mk, 'communication_sizes') / 1024.0
                    enc_t = max(get_mean(size_data[size], scheme, mk, 'encrypt_times'), 0.0)
                    qry_t = max(get_mean(size_data[size], scheme, mk, 'query_times'), 0.0)
                    dec_t = max(get_mean(size_data[size], scheme, mk, 'decrypt_times'), 0.0)
                    total_t = enc_t + qry_t + dec_t
                    if total_t <= 0:
                        ratios = {'encrypt_times': 1.0 / 3.0, 'query_times': 1.0 / 3.0, 'decrypt_times': 1.0 / 3.0}
                    else:
                        ratios = {
                            'encrypt_times': enc_t / total_t,
                            'query_times': qry_t / total_t,
                            'decrypt_times': dec_t / total_t,
                        }
                    for phase_key, _ in PHASES:
                        phase_split_kb[phase_key].append(total_kb * ratios[phase_key])

            comm_by_size[size][scheme] = {
                'phase_kb': {phase_key: float(np.mean(vals)) for phase_key, vals in phase_split_kb.items()},
            }
    return comm_by_size


def add_axis_break_marks(top_ax, bottom_ax):
    kwargs = dict(color='k', clip_on=False, linewidth=1.0)
    top_ax.spines['bottom'].set_visible(False)
    bottom_ax.spines['top'].set_visible(False)
    top_ax.tick_params(labeltop=False)
    bottom_ax.xaxis.tick_bottom()
    top_ax.plot((-0.015, 0.015), (-0.02, 0.02), transform=top_ax.transAxes, **kwargs)
    top_ax.plot((0.985, 1.015), (-0.02, 0.02), transform=top_ax.transAxes, **kwargs)
    bottom_ax.plot((-0.015, 0.015), (0.98, 1.02), transform=bottom_ax.transAxes, **kwargs)
    bottom_ax.plot((0.985, 1.015), (0.98, 1.02), transform=bottom_ax.transAxes, **kwargs)


def draw_communication_chart():
    size_data = load_schemes_for_sizes(SIZE_VALUES)
    comm_by_size = compute_comm_phase_kb(size_data)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        6,
        3,
        height_ratios=[4.0, 1.0, 4.0, 1.0, 4.0, 1.0],
        hspace=0.14,
        wspace=0.22,
    )
    fig.suptitle('Communication Costs: 3x3 Grid', fontsize=14, fontweight='bold')

    legend_handles = None

    for row, size in enumerate(SIZE_VALUES):
        for col, (phase_key, phase_title) in enumerate(PHASES):
            top_ax = fig.add_subplot(gs[row * 2, col])
            bottom_ax = fig.add_subplot(gs[row * 2 + 1, col], sharex=top_ax)

            top_ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            bottom_ax.ticklabel_format(axis='y', style='plain', useOffset=False)

            x = np.arange(len(SCHEMES))
            vals = [comm_by_size[size][scheme]['phase_kb'][phase_key] for scheme in SCHEMES]
            baseline_val = comm_by_size[size]['Naive CCS-2023']['phase_kb'][phase_key]
            other_vals = [comm_by_size[size][scheme]['phase_kb'][phase_key] for scheme in SCHEMES if scheme != 'Naive CCS-2023']

            lower_min = max(0.0, np.floor((baseline_val - 1.0) / 2.0) * 2.0)
            lower_max = np.ceil((baseline_val + 1.0) / 2.0) * 2.0
            other_min = min(other_vals)
            other_max = max(other_vals)
            span = max(other_max - other_min, 1.0)
            margin = max(span * 0.8, 5.0)
            upper_min = max(0.0, np.floor((other_min - margin) / 5.0) * 5.0)
            upper_max = np.ceil((other_max + margin) / 5.0) * 5.0
            if upper_min <= lower_max:
                upper_min = lower_max + 2.0
            if upper_max <= upper_min:
                upper_max = upper_min + 5.0

            bars_top = []
            for i, val in enumerate(vals):
                is_baseline = (SCHEMES[i] == 'Naive CCS-2023')
                face_col = EDGE_COLORS[i] if is_baseline else 'none'
                fill_alpha = 0.45 if is_baseline else 0.99
                lw = 2.3 if is_baseline else 1.2
                bt = top_ax.bar(
                    x[i],
                    val,
                    color=face_col,
                    edgecolor=EDGE_COLORS[i],
                    hatch=HATCHES[i],
                    linewidth=lw,
                    alpha=fill_alpha,
                )
                bottom_ax.bar(
                    x[i],
                    val,
                    color=face_col,
                    edgecolor=EDGE_COLORS[i],
                    hatch=HATCHES[i],
                    linewidth=lw,
                    alpha=fill_alpha,
                )
                bars_top.append(bt[0])

            bottom_ax.set_ylim(lower_min, lower_max)
            top_ax.set_ylim(upper_min, upper_max)
            bottom_ax.set_yticks([lower_min, lower_max])
            bottom_ax.tick_params(axis='y', labelsize=7, length=2)

            top_ax.grid(axis='y', alpha=0.35, linestyle='--', linewidth=0.5, color='black')
            bottom_ax.grid(axis='y', alpha=0.15, linestyle='--', linewidth=0.5, color='black')

            if row == 0:
                top_ax.set_title(phase_title, fontsize=11)
            if col == 0:
                top_ax.set_ylabel(f'Size={size}\nPacket Size (KB)', fontsize=10)

            if row == len(SIZE_VALUES) - 1:
                bottom_ax.set_xticks(x)
                bottom_ax.set_xticklabels(SCHEMES, rotation=15, fontsize=9)
            else:
                bottom_ax.set_xticks(x)
                bottom_ax.set_xticklabels([])

            top_ax.tick_params(labelbottom=False)
            add_axis_break_marks(top_ax, bottom_ax)

            if row == 0 and col == 0 and legend_handles is None:
                legend_handles = bars_top

    fig.text(0.5, 0.04, 'Schemes', ha='center', fontsize=11)
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            SCHEMES,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(SCHEMES),
            frameon=False,
            fontsize=10,
        )

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.88, wspace=0.22, hspace=0.14)
    out = PIC_OUT_DIR / 'communication_6schemes_breakpoint.png'
    plt.savefig(out, dpi=300)
    plt.close()
    print(out)


def draw_communication_chart_no_break_scientific():
    size_data = load_schemes_for_sizes(SIZE_VALUES)
    comm_by_size = compute_comm_phase_kb(size_data)
    sci_scale = 1.0  # Original scale.

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=False, sharey=False)
    fig.suptitle('Communication Costs: Scientific Axis', fontsize=14, fontweight='bold')

    legend_handles = None

    for row, size in enumerate(SIZE_VALUES):
        for col, (phase_key, phase_title) in enumerate(PHASES):
            ax = axes[row, col]

            x = np.arange(len(SCHEMES))
            vals = [comm_by_size[size][scheme]['phase_kb'][phase_key] * sci_scale for scheme in SCHEMES]

            bars = []
            for i, val in enumerate(vals):
                is_baseline = (SCHEMES[i] == 'Naive CCS-2023')
                face_col = EDGE_COLORS[i] if is_baseline else 'none'
                fill_alpha = 0.45 if is_baseline else 0.99
                lw = 2.3 if is_baseline else 1.2
                b = ax.bar(
                    x[i],
                    val,
                    color=face_col,
                    edgecolor=EDGE_COLORS[i],
                    hatch=HATCHES[i],
                    linewidth=lw,
                    alpha=fill_alpha,
                )
                bars.append(b[0])

            ax.grid(axis='y', alpha=0.30, linestyle='--', linewidth=0.5, color='black')

            # Scientific notation on y-axis
            sf = ScalarFormatter(useMathText=True)
            sf.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(sf)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

            # Keep values unchanged; only widen y-axis range so differences look less pronounced.
            ymax = max(vals) if vals else 1.0
            ax.set_ylim(0.0, ymax * 2.5)

            if row == 0:
                ax.set_title(phase_title, fontsize=11)
            if col == 0:
                ax.set_ylabel(f'Size={size}\nPacket Size (KB)', fontsize=10)

            ax.set_xticks(x)
            if row == len(SIZE_VALUES) - 1:
                ax.set_xticklabels(SCHEMES, rotation=15, fontsize=9)
            else:
                ax.set_xticklabels([])

            if row == 0 and col == 0 and legend_handles is None:
                legend_handles = bars

    fig.text(0.5, 0.04, 'Schemes', ha='center', fontsize=11)
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            SCHEMES,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(SCHEMES),
            frameon=False,
            fontsize=10,
        )

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.88, wspace=0.22, hspace=0.28)
    out = PIC_OUT_DIR / 'communication_6schemes_exponential_axis.png'
    plt.savefig(out, dpi=300)
    plt.close()
    print(out)


if __name__ == '__main__':
    draw_communication_chart()
    draw_communication_chart_no_break_scientific()
