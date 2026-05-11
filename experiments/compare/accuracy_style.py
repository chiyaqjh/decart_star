"""Shared plotting style aligned with experiments/results/pic/accuracy.py."""

import matplotlib.pylab as pylab
from pathlib import Path


def get_pic_accuracy_dir(project_root: str) -> Path:
    out = Path(project_root) / 'experiments' / 'results' / 'pic_accuracy'
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
