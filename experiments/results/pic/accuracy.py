# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 01:10:50 2022

@author: 86159
"""

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# 绘图参数全家桶
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '5.4, 2.3',
    'figure.dpi':'300',
    'figure.subplot.left':'0.154',
    'figure.subplot.right':'0.982',
    'figure.subplot.bottom':'0.219',
    'figure.subplot.top':'0.974',
    'pdf.fonttype':'42',
    'ps.fonttype':'42',
}
pylab.rcParams.update(params)

# data3 = [[i for i in data3[d]] for d in range(len(data3))][::-1]

color_1 = "#F27970"
color_2 = "#BB9727"
color_3 = "#54B345"
color_4 = "#32B897"
color_5 = "#05B9E2"

fig, ax1 = plt.subplots()


# 柱状图
data3 = [[0.999385555,	0.999297777,	0.999315333,	0.999332889,	0.999350444],
         [0.9716,	0.9685,	0.969,	0.9695,	0.97025],
         [0.762474923,	0.754142386,	0.754142386,	0.754142386,	0.754142386],
         [0.847032307,	0.817205109,	0.824718257,	0.825845229,	0.826446281]]
x = 1 * np.arange(4)  # the label locations
width = 0.13

ax1.bar(x - 2 * width, [d[0] for d in data3], width, color='none', label='No-privacy PQ', edgecolor=color_1, hatch="-----", alpha=.99)

ax1.bar(x - width, [d[1] for d in data3], width, color='none', label='PEPQ ($10^{-2}$)', edgecolor=color_2, hatch="/////", alpha=.99)

bars = ax1.bar(x, [d[2] for d in data3], width, color='none', label='PEPQ ($10^{-3}$)', edgecolor=color_3, hatch="|||||", alpha=.99)

ax1.bar(x + width, [d[3] for d in data3], width, color='none', label='PEPQ ($10^{-4}$)', edgecolor=color_4, hatch=".....", alpha=.99)

ax1.bar(x + 2 * width, [d[4] for d in data3], width, color='none', label='PEPQ ($10^{-5}$)', edgecolor=color_5, hatch="xxxxx", alpha=.99)

ax1.bar_label(bars, ['-%.2f%%' % ((data3[i][0] - min(data3[i])) * 100) for i in range(4)], padding=1.2)

#折线图
# x = 1 * np.arange(5)  # the label locations
# data3 = [[0.999385555,	0.999297777,	0.999315333,	0.999332889,	0.999350444], # No-privacy PQ
#          [0.9716,	0.9685,	0.969,	0.9695,	0.97025],# PEPQ ($10^{-2}$
#          [0.762474923,	0.754142386,	0.754142386,	0.754142386,	0.754142386],# PEPQ ($10^{-3}$
#          [0.847032307,	0.817205109,	0.824718257,	0.825845229,	0.826446281]]# PEPQ ($10^{-4}$
# plt.plot(x, data3[1], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='No-privacy PQ')
# plt.plot(x, data3[2], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="PEPQ ($10^{-2}$")
# plt.plot(x, data3[3], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="PEPQ ($10^{-3}$")
# plt.plot(x, data3[3], linewidth =2.0, color=color_4, marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,label="PEPQ ($10^{-4}$")

x_ticks = 1 * np.arange(4)
labels = ["Credit card", "Hospital", "Expedia", "Flights"]

ax1.set_xticks(x_ticks)
ax1.set_xticklabels(labels)

ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

ax1.set_xlabel('Dataset')
ax1.set_ylabel('Accuracy')

plt.ylim((0.7, 1.3))

plt.legend(loc='upper left', ncol=3, columnspacing=0.4, prop={'size': 10})

plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.5)

plt.tight_layout()

plt.savefig("./accuracy.pdf", format = 'pdf')

plt.show()