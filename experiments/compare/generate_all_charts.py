# decart/experiments/compare/generate_all_charts.py
"""
生成所有论文图表
"""

import sys
import os
import argparse
import json
import pickle
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

from accuracy_style import apply_accuracy_style, get_pic_accuracy_dir, single_output_path

apply_accuracy_style()

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

PIC_DIR = get_pic_accuracy_dir(project_root)

def get_filename_param_tag() -> str:
    """从最新实验结果提取参数标签，用于输出文件名。"""
    decart_paths = sorted(glob.glob(os.path.join(project_root, 'experiments/results/our_decart/*.json')))
    if not decart_paths:
        return ''

    try:
        data = json.load(open(decart_paths[-1], encoding='utf-8'))
        config = data.get('config', {})
        parts = [
            f"N{config.get('N', 'na')}",
            f"n{config.get('n', 'na')}",
            f"r{config.get('num_records', 'na')}",
            f"d{config.get('record_dim', 'na')}",
            f"p{config.get('policy_size', 'na')}",
            f"runs{config.get('num_runs', 'na')}",
        ]
        return '_'.join(parts)
    except Exception:
        return ''


def save_fig(prefix: str) -> Path:
    """只保留一张图：固定文件名并覆盖旧图。"""
    return single_output_path(PIC_DIR, prefix, 'png')

SCHEME_LABELS = ['DeCart', 'DeCart*', 'CCS23', 'Server', 'Offline']
SCHEME_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
MODEL_LABELS = ['Dot Product', 'Decision Tree', 'Neural Network']
MODEL_KEYS  = ['dot', 'decision_tree', 'neural_network']
SIZE_VALUES = [32, 64, 128]

RESULT_DIRS = {
    'DeCart': 'our_decart',
    'DeCart*': 'our_decart_star',
    'CCS23': 'scheme1_ccs23',
    'Server': 'scheme2_server',
    'Offline': 'scheme3_offline',
}

def load_all_schemes():
    """加载所有方案最新结果"""
    paths = {
        'DeCart':  sorted(glob.glob(os.path.join(project_root, 'experiments/results/our_decart/*.json')))[-1],
        'DeCart*': sorted(glob.glob(os.path.join(project_root, 'experiments/results/our_decart_star/*.json')))[-1],
        'CCS23':   sorted(glob.glob(os.path.join(project_root, 'experiments/results/scheme1_ccs23/*.json')))[-1],
        'Server':  sorted(glob.glob(os.path.join(project_root, 'experiments/results/scheme2_server/*.json')))[-1],
        'Offline': sorted(glob.glob(os.path.join(project_root, 'experiments/results/scheme3_offline/*.json')))[-1],
    }
    data = {}
    for scheme, p in paths.items():
        data[scheme] = json.load(open(p))
    return data


def find_latest_result_by_size(scheme: str, size: int, policy_size: int = 10, num_runs: int = 3):
    """查找某方案在给定规模下最新的结果文件。"""
    result_dir = RESULT_DIRS[scheme]
    paths = sorted(glob.glob(os.path.join(project_root, f'experiments/results/{result_dir}/*.json')), reverse=True)

    for path in paths:
        try:
            data = json.load(open(path, encoding='utf-8'))
        except Exception:
            continue

        config = data.get('config', {})
        if (
            config.get('num_records') == size and
            config.get('record_dim') == size and
            config.get('policy_size') == policy_size and
            config.get('num_runs') == num_runs
        ):
            return data

    raise FileNotFoundError(
        f'未找到 {scheme} 在 num_records=record_dim={size}, policy_size={policy_size}, num_runs={num_runs} 下的结果文件'
    )


def load_schemes_for_sizes(sizes, policy_size: int = 10, num_runs: int = 3):
    """加载多个规模下的所有方案结果。"""
    data = {}
    for size in sizes:
        data[size] = {}
        for scheme in SCHEME_LABELS:
            data[size][scheme] = find_latest_result_by_size(scheme, size, policy_size=policy_size, num_runs=num_runs)
    return data

def get_mean(data, scheme, model, key):
    vals = data[scheme]['models'][model][key]
    return float(np.mean(vals))

# ─────────────────────────────────────────────────────────────
# 图1: 通信开销 5方案对比（各阶段 vs N）
# ─────────────────────────────────────────────────────────────
def plot_communication_5schemes():
    """生成 3x3 图组并使用断轴: 列=阶段，行=参数规模(32/64/128)。"""
    size_data = load_schemes_for_sizes(SIZE_VALUES)
    schemes = SCHEME_LABELS
    colors = SCHEME_COLORS
    phases = [
        ('encrypt_times', 'Encrypt Request Packet'),
        ('query_times', 'Query Request Packet'),
        ('decrypt_times', 'Response Packet'),
    ]

    # 每个规模下优先使用实测阶段通信量；若缺失则回退到按时间占比分摊
    comm_by_size = {}
    for size in SIZE_VALUES:
        comm_by_size[size] = {}
        for scheme in schemes:
            phase_split_kb = {k: [] for k, _ in phases}
            for mk in MODEL_KEYS:
                model_block = size_data[size][scheme]['models'][mk]

                has_measured = (
                    ('comm_upload_sizes' in model_block and len(model_block['comm_upload_sizes']) > 0) and
                    ('comm_query_sizes' in model_block and len(model_block['comm_query_sizes']) > 0) and
                    ('comm_decrypt_sizes' in model_block and len(model_block['comm_decrypt_sizes']) > 0)
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
                    for phase_key, _ in phases:
                        phase_split_kb[phase_key].append(total_kb * ratios[phase_key])

            comm_by_size[size][scheme] = {
                'phase_kb': {phase_key: float(np.mean(vals)) for phase_key, vals in phase_split_kb.items()},
            }

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        6,
        3,
        height_ratios=[4.0, 1.0, 4.0, 1.0, 4.0, 1.0],
        hspace=0.06,
        wspace=0.22,
    )
    fig.suptitle('Communication Costs (Packet Size in KB, Not Time): 3x3 Grid (Phase x Parameter Size, Broken Axis)', fontsize=14, fontweight='bold')

    legend_handles = None
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

    for row, size in enumerate(SIZE_VALUES):
        for col, (phase_key, phase_title) in enumerate(phases):
            top_ax = fig.add_subplot(gs[row * 2, col])
            bottom_ax = fig.add_subplot(gs[row * 2 + 1, col], sharex=top_ax)

            # 禁用科学计数法偏移，避免出现类似 +1.31e5 的轴注释
            top_ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            bottom_ax.ticklabel_format(axis='y', style='plain', useOffset=False)

            x = np.arange(len(schemes))
            vals = [comm_by_size[size][scheme]['phase_kb'][phase_key] for scheme in schemes]
            ccs23_val = comm_by_size[size]['CCS23']['phase_kb'][phase_key]
            other_vals = [comm_by_size[size][scheme]['phase_kb'][phase_key] for scheme in schemes if scheme != 'CCS23']

            lower_min = max(0.0, np.floor((ccs23_val - 1.0) / 2.0) * 2.0)
            lower_max = np.ceil((ccs23_val + 1.0) / 2.0) * 2.0

            other_min = min(other_vals)
            other_max = max(other_vals)
            span = max(other_max - other_min, 1.0)
            margin = max(span * 0.8, 5.0)
            upper_min = max(0.0, np.floor((other_min - margin) / 5.0) * 5.0)
            upper_max = np.ceil((other_max + margin) / 5.0) * 5.0

            # 断轴上半段必须高于下半段，避免两段同时出现 0 刻度
            min_gap = 2.0
            if upper_min <= lower_max:
                upper_min = lower_max + min_gap
            if upper_max <= upper_min:
                upper_max = upper_min + 5.0

            bars_top = top_ax.bar(x, vals, color=colors, alpha=0.88, edgecolor='white')
            bottom_ax.bar(x, vals, color=colors, alpha=0.88, edgecolor='white')
            for bar_obj, scheme_name in zip(bars_top, schemes):
                bar_obj.set_label(scheme_name)

            bottom_ax.set_ylim(lower_min, lower_max)
            top_ax.set_ylim(upper_min, upper_max)
            bottom_ax.set_yticks([lower_min, lower_max])
            bottom_ax.tick_params(axis='y', labelsize=7, length=2)

            top_ax.grid(axis='y', alpha=0.35)
            bottom_ax.grid(axis='y', alpha=0.12)

            if row == 0:
                top_ax.set_title(phase_title, fontsize=11)
            if col == 0:
                top_ax.set_ylabel(f'Size={size}\nPacket Size (KB)', fontsize=10)

            if row == len(SIZE_VALUES) - 1:
                bottom_ax.set_xticks(x)
                bottom_ax.set_xticklabels(schemes, rotation=15, fontsize=9)
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
            schemes,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(schemes),
            frameon=False,
            fontsize=9,
        )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.88, wspace=0.22, hspace=0.06)
    out = save_fig('communication_5schemes_sizes32_64_128')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✓ 图1保存: {out.name}")


# ─────────────────────────────────────────────────────────────
# 图2: 总耗时堆叠图（归一化到 CCS23=1，Encrypt+Query+Decrypt）
# ─────────────────────────────────────────────────────────────
def plot_stacked_total_time():
    data = load_all_schemes()
    schemes = SCHEME_LABELS
    colors_enc  = ['#1565C0', '#E65100', '#1B5E20', '#4A148C', '#B71C1C']
    colors_qry  = ['#42A5F5', '#FFA726', '#66BB6A', '#AB47BC', '#EF5350']
    colors_dec  = ['#90CAF9', '#FFCC80', '#A5D6A7', '#CE93D8', '#EF9A9A']

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
    fig.suptitle('End-to-End Latency Breakdown\n(Normalized to CCS23 = 1, higher = slower)', fontsize=13, fontweight='bold')

    for ax_idx, (model_key, model_label) in enumerate(zip(MODEL_KEYS, MODEL_LABELS)):
        ax = axes[ax_idx]
        x = np.arange(len(schemes))
        width = 0.55

        # 以 CCS23 总时间为基准做归一化
        ccs23_total = (get_mean(data, 'CCS23', model_key, 'encrypt_times') +
                       get_mean(data, 'CCS23', model_key, 'query_times') +
                       get_mean(data, 'CCS23', model_key, 'decrypt_times'))

        enc_vals = [get_mean(data, s, model_key, 'encrypt_times') / ccs23_total for s in schemes]
        qry_vals = [get_mean(data, s, model_key, 'query_times')   / ccs23_total for s in schemes]
        dec_vals = [get_mean(data, s, model_key, 'decrypt_times') / ccs23_total for s in schemes]

        ax.bar(x, enc_vals, width,
               color=[colors_enc[i] for i in range(len(schemes))], alpha=0.9)
        ax.bar(x, qry_vals, width, bottom=enc_vals,
               color=[colors_qry[i] for i in range(len(schemes))], alpha=0.9)
        ax.bar(x, dec_vals, width,
               bottom=[e+q for e,q in zip(enc_vals, qry_vals)],
               color=[colors_dec[i] for i in range(len(schemes))], alpha=0.9)

        # 在柱顶标注总倍数
        for xi, (e, q, d) in enumerate(zip(enc_vals, qry_vals, dec_vals)):
            total = e + q + d
            ax.text(xi, total * 1.05, f'{total:.0f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # CCS23=1 基准线
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1.2, label='CCS23 baseline')

        ax.set_title(model_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(schemes, fontsize=9, rotation=15)
        ax.set_ylabel('Normalized Latency (× CCS23)', fontsize=10)
        ax.grid(axis='y', alpha=0.4)

        if ax_idx == 0:
            enc_patch = mpatches.Patch(color='#555555', label='Encrypt (dark)')
            qry_patch = mpatches.Patch(color='#888888', label='Query (mid)')
            dec_patch = mpatches.Patch(color='#AAAAAA', label='Decrypt (light)')
            ax.legend(handles=[enc_patch, qry_patch, dec_patch,
                                plt.Line2D([0],[0], color='black', linestyle='--', label='CCS23=1')],
                      fontsize=8)

    plt.tight_layout()
    out = save_fig('stacked_latency')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图2保存: {out.name}")


# ─────────────────────────────────────────────────────────────
# 图3: 通信量5方案柱状图（归一化到 CCS23=1）
# ─────────────────────────────────────────────────────────────
def plot_communication_bar():
    data = load_all_schemes()
    schemes = SCHEME_LABELS
    colors  = SCHEME_COLORS

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[4.0, 1.0], hspace=0.06, wspace=0.22)
    fig.suptitle('Communication Cost Comparison\n(Normalized to CCS23 = 1, broken axis ignores non-informative middle range)',
                 fontsize=13, fontweight='bold')

    def total_comm_mean(scheme_name, scheme_data, model_key):
        model_block = scheme_data['models'][model_key]
        has_measured = (
            ('comm_upload_sizes' in model_block and len(model_block['comm_upload_sizes']) > 0) and
            ('comm_query_sizes' in model_block and len(model_block['comm_query_sizes']) > 0) and
            ('comm_decrypt_sizes' in model_block and len(model_block['comm_decrypt_sizes']) > 0)
        )
        if has_measured:
            return (
                float(np.mean(model_block['comm_upload_sizes'])) +
                float(np.mean(model_block['comm_query_sizes'])) +
                float(np.mean(model_block['comm_decrypt_sizes']))
            )
        return get_mean(data, scheme_name, model_key, 'communication_sizes')

    def format_ratio_label(val: float) -> str:
        """提高显示精度，避免大数倍率被过度取整。"""
        if val >= 1:
            return f'{val:.2f}×'
        if val >= 0.1:
            return f'{val:.3f}×'
        return f'{val:.4f}×'

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

    for col, (model_key, model_label) in enumerate(zip(MODEL_KEYS, MODEL_LABELS)):
        top_ax = fig.add_subplot(gs[0, col])
        bottom_ax = fig.add_subplot(gs[1, col], sharex=top_ax)

        # CCS23 基准通信量
        ccs23_comm = max(total_comm_mean('CCS23', data['CCS23'], model_key), 1e-12)
        norm_vals = [max(total_comm_mean(s, data[s], model_key) / ccs23_comm, 0.0) for s in schemes]
        x = np.arange(len(schemes))

        other_vals = [v for s, v in zip(schemes, norm_vals) if s != 'CCS23']
        lower_min = 0.0
        lower_max = max(2.0, 1.6)

        upper_min = max(lower_max + 0.2, min(other_vals) * 0.9)
        upper_max = max(upper_min + 1.0, max(other_vals) * 1.1)

        bars_top = top_ax.bar(x, norm_vals, color=colors, alpha=0.85, edgecolor='white')
        bottom_ax.bar(x, norm_vals, color=colors, alpha=0.85, edgecolor='white')

        top_ax.set_ylim(upper_min, upper_max)
        bottom_ax.set_ylim(lower_min, lower_max)
        bottom_ax.set_yticks([0.0, 1.0, lower_max])

        top_ax.axhline(y=1, color='black', linestyle='--', linewidth=1.2, label='CCS23 baseline')
        bottom_ax.axhline(y=1, color='black', linestyle='--', linewidth=1.2)

        # 标注倍数
        for bar, scheme_name, val in zip(bars_top, schemes, norm_vals):
            xpos = bar.get_x() + bar.get_width() / 2
            label = format_ratio_label(val)

            if scheme_name == 'CCS23':
                bottom_ax.text(xpos, max(val * 1.04, 1.05), label,
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                top_ax.text(xpos, val * 1.01, label,
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        top_ax.set_title(model_label, fontsize=12)
        top_ax.set_ylabel('Normalized Comm. (× CCS23)', fontsize=10)
        top_ax.grid(axis='y', alpha=0.35)
        bottom_ax.grid(axis='y', alpha=0.12)

        bottom_ax.set_xticks(x)
        bottom_ax.set_xticklabels(schemes, rotation=15)
        top_ax.tick_params(labelbottom=False)
        add_axis_break_marks(top_ax, bottom_ax)

        if col == 0:
            top_ax.legend(fontsize=8, loc='upper left')
    fig.text(0.5, 0.04, 'Schemes', ha='center', fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.85, wspace=0.22, hspace=0.06)
    out = save_fig('communication_bar')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图3保存: {out.name}")


# ─────────────────────────────────────────────────────────────
# 图4: 吞吐量对比图（归一化到 CCS23=1）
# ─────────────────────────────────────────────────────────────
def plot_throughput():
    data = load_all_schemes()
    schemes = SCHEME_LABELS
    colors  = SCHEME_COLORS

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('System Throughput Comparison\n(Normalized to CCS23 = 1, lower = less throughput)',
                 fontsize=13, fontweight='bold')

    for ax, model_key, model_label in zip(axes, MODEL_KEYS, MODEL_LABELS):
        def throughput(scheme):
            total_t = (get_mean(data, scheme, model_key, 'encrypt_times') +
                       get_mean(data, scheme, model_key, 'query_times') +
                       get_mean(data, scheme, model_key, 'decrypt_times'))
            return 1.0 / total_t if total_t > 0 else 0

        ccs23_tp = throughput('CCS23')
        norm_vals = [throughput(s) / ccs23_tp for s in schemes]

        bars = ax.bar(schemes, norm_vals, color=colors, alpha=0.85, edgecolor='white')
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1.2, label='CCS23 baseline')

        for bar, val in zip(bars, norm_vals):
            label = f'{val:.3f}×' if val < 0.1 else f'{val:.2f}×'
            ax.text(bar.get_x() + bar.get_width()/2,
                    max(bar.get_height(), 0.001) * 1.08,
                    label, ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(model_label, fontsize=12)
        ax.set_ylabel('Normalized Throughput (× CCS23)', fontsize=10)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.4)
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = save_fig('throughput')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图4保存: {out.name}")


# ─────────────────────────────────────────────────────────────
# 图5: 存储分项堆叠图（DeCart vs DeCart* 各组件 vs N）
# ─────────────────────────────────────────────────────────────
def plot_storage_breakdown():
    pkl_files = sorted(glob.glob(os.path.join(project_root, 'experiments/results/storage_benchmark_dynamic/*.pkl')))
    if not pkl_files:
        print("  [跳过] 无存储基准测试数据")
        return

    raw = pickle.load(open(pkl_files[-1], 'rb'))
    N_values = raw['decart']['N_values']
    components = ['crs', 'pp', 'aux', 'user_secrets', 'trust_map', 'policies', 'encrypted_data']
    comp_labels = ['CRS', 'Public Params', 'Auxiliary', 'User Secrets', 'Trust Map', 'Policies', 'Encrypted Data']
    comp_colors = ['#F27970', '#BB9727', '#54B345', '#32B897', '#05B9E2', '#3A5A98', '#E07A5F']
    comp_hatches = ['-----', '/////', '|||||', '.....', 'xxxxx', '+++++', 'ooooo']

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4), sharey=True)
    fig.suptitle('Storage Breakdown: DeCart vs DeCart* (KB)', fontsize=13, fontweight='bold', y=0.98)

    legend_handles = [
        mpatches.Patch(facecolor='none', edgecolor=c, hatch=h, label=lbl, linewidth=1.2)
        for lbl, c, h in zip(comp_labels, comp_colors, comp_hatches)
    ]

    for ax, scheme, title in zip(axes, ['decart', 'decart_star'], ['DeCart (O(n²))', 'DeCart* (O(n))']):
        sizes = raw[scheme]['sizes']
        bottoms = np.zeros(len(N_values))
        totals = np.zeros(len(N_values))

        for comp, label, color, hatch in zip(components, comp_labels, comp_colors, comp_hatches):
            vals = np.array([s[comp] for s in sizes])
            ax.bar(
                range(len(N_values)),
                vals,
                bottom=bottoms,
                label=label,
                color='none',
                edgecolor=color,
                hatch=hatch,
                linewidth=1.2,
                alpha=0.95,
            )

            bottoms += vals
            totals += vals

        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(N_values)))
        ax.set_xticklabels([str(n) for n in N_values])
        ax.set_xlabel('N (Max Users)', fontsize=11)
        ax.set_ylabel('Storage (KB)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        frameon=False,
        fontsize=10,
        columnspacing=0.8,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.80])
    out = save_fig('storage_breakdown')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图5保存: {out.name}")


# ─────────────────────────────────────────────────────────────
# 图6: 加速比热力图（DeCart / baseline, DeCart* / baseline）
# ─────────────────────────────────────────────────────────────
def plot_speedup_heatmap():
    data = load_all_schemes()
    metrics   = ['encrypt_times', 'query_times', 'decrypt_times']
    m_labels  = ['Encrypt', 'Query', 'Decrypt']
    baselines = ['Server', 'Offline']

    def format_ratio(val: float) -> str:
        """避免把极小非零值误显示为 0.0x。"""
        if val == 0:
            return '0x'
        if val < 0.001:
            return f'{val:.1e}x'
        if val < 0.1:
            return f'{val:.3f}x'
        if val < 10:
            return f'{val:.2f}x'
        return f'{val:.1f}x'

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Speedup Ratio Heatmap (Baseline / Our Scheme, CCS23 Removed)', fontsize=13, fontweight='bold')

    for row_idx, our_scheme in enumerate(['DeCart', 'DeCart*']):
        for col_idx, baseline in enumerate(baselines):
            ax = axes[row_idx][col_idx]
            matrix = np.full((len(MODEL_LABELS), len(m_labels)), np.nan)

            for mi, mk in enumerate(MODEL_KEYS):
                for ti, tk in enumerate(metrics):
                    our_val  = get_mean(data, our_scheme, mk, tk)
                    base_val = get_mean(data, baseline, mk, tk)
                    if base_val <= 0 or our_val <= 0:
                        # 基线或本方案该阶段耗时为 0 时，倍率在语义上不可比
                        matrix[mi][ti] = np.nan
                    else:
                        matrix[mi][ti] = base_val / our_val

            valid_vals = matrix[np.isfinite(matrix)]
            vmax = max(1.0, float(valid_vals.max())) if valid_vals.size > 0 else 1.0
            cmap = plt.cm.get_cmap('RdYlGn_r').copy()
            cmap.set_bad(color='#D9D9D9')
            im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
            ax.set_xticks(range(len(m_labels)))
            ax.set_xticklabels(m_labels, fontsize=9)
            ax.set_yticks(range(len(MODEL_LABELS)))
            ax.set_yticklabels(MODEL_LABELS, fontsize=9)
            ax.set_title(f'{our_scheme} vs {baseline}', fontsize=10, fontweight='bold')

            for mi in range(len(MODEL_LABELS)):
                for ti in range(len(m_labels)):
                    val = matrix[mi][ti]
                    if np.isfinite(val):
                        text_color = 'white' if val > vmax * 0.6 else 'black'
                        label = format_ratio(val)
                    else:
                        text_color = 'black'
                        label = 'N/A'
                    ax.text(ti, mi, label, ha='center', va='center',
                            fontsize=9, color=text_color, fontweight='bold')

            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out = save_fig('speedup_heatmap')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 图6保存: {out.name}")


# ─────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成论文图表')
    parser.add_argument(
        '--only',
        choices=['all', 'comm', 'stacked', 'bar', 'throughput', 'storage', 'heatmap'],
        default='all',
        help='仅生成指定图表: all|comm|stacked|bar|throughput|storage|heatmap'
    )
    args = parser.parse_args()

    jobs = [
        ('comm', '通信开销 5方案对比图', plot_communication_5schemes),
        ('stacked', '总耗时堆叠图', plot_stacked_total_time),
        ('bar', '通信量柱状图', plot_communication_bar),
        ('throughput', '吞吐量对比图', plot_throughput),
        ('storage', '存储分项堆叠图', plot_storage_breakdown),
        ('heatmap', '加速比热力图', plot_speedup_heatmap),
    ]

    selected_jobs = jobs if args.only == 'all' else [j for j in jobs if j[0] == args.only]

    print("=" * 55)
    if args.only == 'all':
        print("生成所有图表")
    else:
        print(f"仅生成图表: {args.only}")
    print("=" * 55)

    total = len(selected_jobs)
    for idx, (_, title, fn) in enumerate(selected_jobs, start=1):
        print(f"\n[{idx}/{total}] {title}...")
        fn()

    print(f"\n图表已保存至: {PIC_DIR}")
