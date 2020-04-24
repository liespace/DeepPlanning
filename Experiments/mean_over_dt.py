from __future__ import print_function
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import visual
import csv

amount = 1725.
threshold = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
c23_cp_xm = np.array([0.126475, 0.023975, 0.144231, 0.236411, 0.301616, 0.381419, 0.487708, 0.621407, 0.764152])
c23_cp_ym = np.array([0.061614, 0.034455, 0.039238, 0.04668, 0.032794, 0.028196, 0.000263, 0.015227, 0.02977])
c23_cp_tm = np.array([0.013306, 0.010019, 0.006302, 0.00466, 0.006555, 0.007683, 0.001037, 0.005474, 0.008213])

c12_cp_xm = np.array([0.694028, 0.304491, 0.051959, 0.148779, 0.315377, 0.482834, 0.681032, 0.954264, 1.245116])
c12_cp_ym = np.array([0.102511, 0.139909, 0.15537, 0.108481, 0.111919, 0.126384, 0.12873, 0.173781, 0.158282])
c12_cp_tm = np.array([0.022667, 0.010289, 0.007581, 0.011014, 0.009649, 0.010052, 0.005315, 0.005401, 0.014489])

plt.rcParams["font.family"] = "Times New Roman"
lines = ([1, 0],  [6, 2], [4, 2, 1, 2], [4, 2, 1, 1, 1, 2], [3, 2], [1, 1])


def mean_over_dt():
    # accuracy over threshold
    fontsize = 50
    bar_width = 0.025
    alpha = 0.8
    color = ['g', 'r', 'b']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('ratio [-]', fontsize=fontsize)
    ax1.set_xlabel('discrimination threshold[-]', fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax1.set_xticks(np.arange(0, 1, 0.1))
    # ax1.set_ylim(bottom=0, top=3.0)
    # ax1.set_yticks(np.arange(-100, 1, 1))
    l1 = ax1.bar(
        threshold-bar_width, c23_cp_xm / c12_cp_xm,
        width=bar_width,
        color=color[0],
        alpha=alpha,
        align='center')
    l2 = ax1.bar(
        threshold, c23_cp_ym / c12_cp_ym,
        width=bar_width,
        color=color[1],
        alpha=alpha,
        align='center')
    l3 = ax1.bar(
        threshold+bar_width, c23_cp_tm / c12_cp_tm,
        width=bar_width,
        color=color[2],
        alpha=alpha,
        align='center')
    ax1.axhline(y=1, linewidth=6, linestyle='--', color='r')
    # ax1.scatter(threshold, c12_np/ amount, marker='D',
    #             s=300, c=color[0], zorder=50)
    # ax1.scatter(threshold, c23_np/ amount, marker='X',
    #             s=400, c=color[1], zorder=100)
    # ax1.scatter(threshold, t28_np/ amount, marker='^',
    #             s=400, c=color[2], zorder=0)

    plt.legend([l1, l2, l3],
               [r'$X$-error Mean', r'$Y$-error Mean', r'$\theta$-error Mean'],
               bbox_to_anchor=(0.5, 0.5), prop={'size': 40})


if __name__ == '__main__':
    mean_over_dt()
    plt.show()
