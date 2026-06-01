from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from accuracy_style import SCHEME_STYLES, apply_accuracy_style, style_axes


ROOT = Path(__file__).resolve().parents[2]
DECART_DIR = ROOT / "experiments" / "results" / "data_new" / "our_decart" / "revoke"
DECART_STAR_DIR = ROOT / "experiments" / "results" / "data_new" / "our_decart_star" / "revoke"
OUTPUT_DIR = ROOT / "experiments" / "results" / "pic_new" / "revoke"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def latest_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {directory / pattern}")
    return matches[-1]


def owner_update_ms(result: dict) -> float:
    return float(result["runs"][0]["owner_update_time"]) * 1000.0


def revoke_ms(result: dict) -> float:
    return float(result["runs"][0]["revoke_time"]) * 1000.0


def plot_line(x_values: list[int], decart_values: list[float], decart_star_values: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    decart_style = SCHEME_STYLES['DeCart']
    decart_star_style = SCHEME_STYLES['DeCart*']
    ax.plot(
        x_values,
        decart_values,
        marker=decart_style['marker'],
        linewidth=2.2,
        markersize=7,
        color=decart_style['edgecolor'],
        markerfacecolor=decart_style['facecolor'],
        markeredgecolor=decart_style['edgecolor'],
        markeredgewidth=1.6,
        label='DeCart',
    )
    ax.plot(
        x_values,
        decart_star_values,
        marker=decart_star_style['marker'],
        linewidth=2.2,
        markersize=7,
        color=decart_star_style['edgecolor'],
        markerfacecolor=decart_star_style['facecolor'],
        markeredgecolor=decart_star_style['edgecolor'],
        markeredgewidth=1.6,
        label='DeCart*',
    )
    ax.set_xlabel("n")
    ax.set_ylabel("Revoke time (ms)")
    ax.set_xticks(x_values)
    ax.set_axisbelow(True)
    ax.margins(x=0.05)
    style_axes(ax, grid_axis='y')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars(
    x_values: list[int],
    decart_values: list[float],
    decart_star_values: list[float],
    output_path: Path,
    x_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    positions = np.arange(len(x_values), dtype=float)
    width = 0.36
    decart_style = SCHEME_STYLES['DeCart']
    decart_star_style = SCHEME_STYLES['DeCart*']

    ax.bar(
        positions - width / 2,
        decart_values,
        width=width,
        label='DeCart',
        color=decart_style['facecolor'],
        edgecolor=decart_style['edgecolor'],
        linewidth=1.4,
        hatch=decart_style['hatch'],
        zorder=3,
    )
    ax.bar(
        positions + width / 2,
        decart_star_values,
        width=width,
        label='DeCart*',
        color=decart_star_style['facecolor'],
        edgecolor=decart_star_style['edgecolor'],
        linewidth=1.4,
        hatch=decart_star_style['hatch'],
        zorder=3,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Owner update time (ms)")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_axisbelow(True)
    ax.margins(x=0.04)
    style_axes(ax, grid_axis='y')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_revoke_time_vs_n() -> None:
    n_values = [8, 16, 32, 64, 128, 256]
    decart_values = []
    decart_star_values = []

    for n in n_values:
        decart_path = latest_match(
            DECART_DIR,
            f"decart_revoke_N10000_n{n}_records10_dim10_policy32_revoke1_decision_tree_*.json",
        )
        decart_star_path = latest_match(
            DECART_STAR_DIR,
            f"decart_star_revoke_N10000_n{n}_records10_dim10_policy32_revoke1_decision_tree_*.json",
        )
        decart_values.append(revoke_ms(load_json(decart_path)))
        decart_star_values.append(revoke_ms(load_json(decart_star_path)))

    plot_line(n_values, decart_values, decart_star_values, OUTPUT_DIR / "revoke_time_vs_n.png")


def build_owner_update_vs_policy_size() -> None:
    policy_sizes = [8, 16, 32, 64, 128]
    decart_values = []
    decart_star_values = []

    for policy_size in policy_sizes:
        decart_path = latest_match(
            DECART_DIR,
            f"decart_revoke_N10000_n32_records10_dim10_policy{policy_size}_revoke1_decision_tree_*.json",
        )
        decart_star_path = latest_match(
            DECART_STAR_DIR,
            f"decart_star_revoke_N10000_n32_records10_dim10_policy{policy_size}_revoke1_decision_tree_*.json",
        )
        decart_values.append(owner_update_ms(load_json(decart_path)))
        decart_star_values.append(owner_update_ms(load_json(decart_star_path)))

    plot_grouped_bars(
        policy_sizes,
        decart_values,
        decart_star_values,
        OUTPUT_DIR / "owner_update_time_vs_policy_size.png",
        x_label="policy_size",
    )


def build_owner_update_vs_revoked_user_count() -> None:
    revoked_user_counts = [1, 2, 4, 8, 16]
    decart_values = []
    decart_star_values = []

    for revoked_user_count in revoked_user_counts:
        decart_path = latest_match(
            DECART_DIR,
            f"decart_revoke_N10000_n32_records10_dim10_policy32_revoke{revoked_user_count}_decision_tree_*.json",
        )
        decart_star_path = latest_match(
            DECART_STAR_DIR,
            f"decart_star_revoke_N10000_n32_records10_dim10_policy32_revoke{revoked_user_count}_decision_tree_*.json",
        )
        decart_values.append(owner_update_ms(load_json(decart_path)))
        decart_star_values.append(owner_update_ms(load_json(decart_star_path)))

    plot_grouped_bars(
        revoked_user_counts,
        decart_values,
        decart_star_values,
        OUTPUT_DIR / "owner_update_time_vs_revoked_user_count.png",
        x_label="revoked_user_count",
    )


def main() -> None:
    apply_accuracy_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_revoke_time_vs_n()
    build_owner_update_vs_policy_size()
    build_owner_update_vs_revoked_user_count()
    print(f"Saved revoke plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()