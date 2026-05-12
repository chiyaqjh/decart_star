import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[3]
compare_dir = project_root / 'experiments' / 'compare'
if str(compare_dir) not in sys.path:
    sys.path.insert(0, str(compare_dir))

from accuracy_style import apply_accuracy_style

apply_accuracy_style()

users = [32, 64, 128, 256, 512]
labels = ['DeCart', 'DeCart*', 'CCS23', 'Server', 'Offline']
data = {
    'DeCart': [12500, 24000, 47000, 93000, 186000],
    'DeCart*': [12200, 23800, 46800, 92500, 185000],
    'CCS23': [320, 340, 360, 380, 400],
    'Server': [12400, 24100, 47200, 93200, 186500],
    'Offline': [12350, 23950, 46950, 92850, 185500],
}

bar_width = 0.14
x = np.arange(len(users))

fig, ax = plt.subplots(figsize=(10, 6))
scheme_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
fill_colors = ['#90CAF9', '#FFE0B2', '#C8E6C9', '#E1BEE7', '#FFCDD2']
hatches = ['//', 'xx', '..', 'oo', '++']

for i, label in enumerate(labels):
    ax.bar(
        x + (i - 2) * bar_width,
        data[label],
        bar_width,
        label=label,
        color=fill_colors[i],
        edgecolor=scheme_colors[i],
        linewidth=1.5,
        hatch=hatches[i],
        zorder=3,
    )

ax.set_xlabel('Total number of users N', fontsize=14)
ax.set_ylabel('Communication (KB)', fontsize=14)
ax.set_title('Communication vs Total Users N', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(users, fontsize=12)

from matplotlib.ticker import AutoMinorLocator
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(which='major', axis='y', linestyle='--', linewidth=1, color='#888', alpha=0.6, zorder=0)
ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.7, color='#bbb', alpha=0.4, zorder=0)
ax.legend(fontsize=12, frameon=True, edgecolor='#ccc')

plt.tight_layout()
output_path = project_root / 'experiments' / 'results' / 'pic_accuracy' / '通信' / 'communication_vs_N_bar.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'Bar chart saved to {output_path}')