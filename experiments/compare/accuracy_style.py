"""Shared plotting style aligned with experiments/results/pic/accuracy.py."""

import matplotlib.pylab as pylab
from pathlib import Path


SCHEME_STYLES = {
    'DeCart': {
        'edgecolor': '#339AF0',
        'facecolor': '#A9D6FF',
        'hatch': '////',
        'marker': 'o',
    },
    'DeCart*': {
        'edgecolor': '#FF9800',
        'facecolor': '#FFE0B2',
        'hatch': '....',
        'marker': 's',
    },
}


def get_pic_accuracy_dir(project_root: str) -> Path:
    out = Path(project_root) / 'experiments' / 'results' / 'pic_new'
    out.mkdir(parents=True, exist_ok=True)
    return out


def single_output_path(output_dir: Path, prefix: str, ext: str = 'png') -> Path:
    for p in output_dir.glob(f'{prefix}_*.{ext}'):
        p.unlink(missing_ok=True)
    fixed = output_dir / f'{prefix}.{ext}'
    fixed.unlink(missing_ok=True)
    return fixed


def apply_accuracy_style():
    params = {
        'axes.labelsize': '14',
        'xtick.labelsize': '12',
        'ytick.labelsize': '12',
        'legend.fontsize': '12',
        'figure.dpi': '300',
        'pdf.fonttype': '42',
        'ps.fonttype': '42',
    }
    pylab.rcParams.update(params)


def style_axes(ax, grid_axis: str = 'y'):
    ax.grid(
        True,
        axis=grid_axis,
        linestyle='--',
        linewidth=0.8,
        alpha=0.3,
        color='#8A8A8A',
    )
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
