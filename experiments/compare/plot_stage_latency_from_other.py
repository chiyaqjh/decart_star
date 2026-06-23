"""Plot Setup/KeyGen/Register latency vs n from data_new/other JSON files."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OTHER_DIRS = {
    "DeCart": PROJECT_ROOT / "experiments" / "results" / "data_new" / "our_decart" / "other",
    "DeCart*": PROJECT_ROOT / "experiments" / "results" / "data_new" / "our_decart_star" / "other",
}
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "pic_new" / "computation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_VALUE = 10000
NUM_RECORDS = 10
RECORD_DIM = 10
POLICY_SIZE = 32
TARGET_NS = [16, 32, 64, 128, 256, 512]

SCHEME_STYLES = {
    "DeCart": {"edgecolor": "#2196F3", "facecolor": "#90CAF9", "hatch": "///"},
    "DeCart*": {"edgecolor": "#FF9800", "facecolor": "#FFE0B2", "hatch": "..."},
}

METRICS = [
    ("setup_time", "n_latency_setup.png", "Setup"),
    ("keygen_time", "n_latency_keygen.png", "KeyGen"),
    ("register_time", "n_latency_register.png", "Register"),
]


def load_latest_for_n(folder: Path, n_value: int) -> dict | None:
    candidates: list[tuple[str, dict]] = []
    for path_str in glob.glob(str(folder / "*pap_id_n_data_*.json")):
        path = Path(path_str)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
        except Exception:
            continue

        cfg = doc.get("config", {})
        n_values = cfg.get("n_values") or []
        if len(n_values) != 1 or int(n_values[0]) != n_value:
            continue

        results = doc.get("results") or []
        if not results:
            continue

        item = results[0]
        item_cfg = item.get("config", {})
        if int(item_cfg.get("N", -1)) != N_VALUE:
            continue
        if int(item_cfg.get("num_records", -1)) != NUM_RECORDS:
            continue
        if int(item_cfg.get("record_dim", -1)) != RECORD_DIM:
            continue
        if int(item_cfg.get("policy_size", -1)) != POLICY_SIZE:
            continue

        candidates.append((path.name, item))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def collect_metric(metric_key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {scheme: [] for scheme in OTHER_DIRS}
    for n_value in TARGET_NS:
        for scheme, folder in OTHER_DIRS.items():
            item = load_latest_for_n(folder, n_value)
            if item is None:
                out[scheme].append(float("nan"))
                continue
            summary = item.get("summary", {})
            out[scheme].append(float(summary.get(metric_key, float("nan"))) * 1000.0)
    return out


def style_axes(ax: plt.Axes) -> None:
    ax.grid(which="major", axis="y", linestyle="--", linewidth=1.0, color="#999", alpha=0.7, zorder=0)


def plot_metric(metric_key: str, out_name: str, title_tag: str) -> None:
    data = collect_metric(metric_key)
    schemes = ["DeCart", "DeCart*"]

    x = np.arange(len(TARGET_NS), dtype=float)
    width = 0.28
    fig, ax = plt.subplots(figsize=(11.0, 6.2))

    for idx, scheme in enumerate(schemes):
        style = SCHEME_STYLES[scheme]
        ax.bar(
            x + (idx - 0.5) * width,
            np.array(data[scheme], dtype=float),
            width=width,
            label=scheme,
            color=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=2.0,
            hatch=style["hatch"],
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in TARGET_NS], fontsize=12)
    ax.set_xlabel("Block size n", fontsize=16)
    if metric_key == "setup_time":
        # Setup spans several orders of magnitude; log scale keeps both schemes visible.
        ax.set_yscale("log")
        ax.set_ylabel("Running time (ms, log scale)", fontsize=16)
    else:
        ax.set_ylabel("Running time (ms)", fontsize=16)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.83), frameon=True, edgecolor="#bbb", fontsize=12)
    style_axes(ax)

    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    for metric_key, out_name, title_tag in METRICS:
        plot_metric(metric_key, out_name, title_tag)


if __name__ == "__main__":
    main()