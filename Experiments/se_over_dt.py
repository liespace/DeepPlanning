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
threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
c23_cp_xm = np.array([2.320599, 2.240394, 2.219677, 2.174547, 2.151044, 2.177014, 2.166137, 2.173105, 2.211882])
c23_cp_ym = np.array([2.040799, 2.017456, 2.049601, 2.039882, 2.040058, 2.0593, 2.094497, 2.124551, 2.165087])
c23_cp_tm = np.array([0.497552, 0.475088, 0.450156, 0.452682, 0.452187, 0.457299, 0.461365, 0.4653, 0.470376])

c12_cp_xm = np.array([2.630387, 2.409868, 2.417857, 2.379078, 2.326274, 2.297216, 2.307799, 2.350501, 2.416509])
c12_cp_ym = np.array([2.107071, 2.108553, 2.079307, 2.132014, 2.136471, 2.133073, 2.161077, 2.200224, 2.273075])
c12_cp_tm = np.array([0.595526, 0.565854, 0.532739, 0.496728, 0.493042, 0.473364, 0.475275, 0.494431, 0.517839])

plt.rcParams["font.family"] = "Times New Roman"
lines = ([1, 0],  [6, 2], [3, 2], [1, 1], [4, 2, 1, 2], [4, 2, 1, 1, 1, 2])


def mean_over_dt():
    # accuracy over threshold
    fontsize = 54
    linewidth = 6
    color = ['r', 'b', 'g']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('accuracy[-]', fontsize=fontsize)
    ax1.set_xlabel('discrimination threshold[-]', fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax1.set_xticks(np.arange(0, 1, 0.1))
    # ax1.set_ylim(bottom=-18, top=1)
    # ax1.set_yticks(np.arange(-100, 1, 1))
    # ax1.set_yticks(yt1)
    l1, = ax1.plot(threshold, c23_cp_xm,
                   linewidth=linewidth,
                   color=color[0], zorder=50,
                   dashes=lines[0])
    l2, = ax1.plot(threshold, c12_cp_xm,
                   linewidth=linewidth,
                   color=color[0], zorder=100,
                   dashes=lines[1])
    l3, = ax1.plot(threshold, c23_cp_ym,
                   linewidth=linewidth,
                   color=color[1], zorder=0,
                   dashes=lines[2])
    l4, = ax1.plot(threshold, c12_cp_ym,
                   linewidth=linewidth,
                   color=color[1], zorder=50,
                   dashes=lines[3])
    l5, = ax1.plot(threshold, c23_cp_tm,
                   linewidth=linewidth,
                   color=color[2], zorder=100,
                   dashes=lines[4])
    l6, = ax1.plot(threshold, c12_cp_tm,
                   linewidth=linewidth,
                   color=color[2], zorder=0,
                   dashes=lines[5])
    # ax1.scatter(threshold, c12_np/ amount, marker='D',
    #             s=300, c=color[0], zorder=50)
    # ax1.scatter(threshold, c23_np/ amount, marker='X',
    #             s=400, c=color[1], zorder=100)
    # ax1.scatter(threshold, t28_np/ amount, marker='^',
    #             s=400, c=color[2], zorder=0)

    plt.legend([l1, l2, l3, l4, l5, l6],
               ['YIPS-V-8-XM', 'YIPS-V-6-XM',
                'YIPS-V-8-YM', 'YIPS-V-6-YM',
                'YIPS-V-8-TM', 'YIPS-V-6-TM'],
               bbox_to_anchor=(0.95, 0.74), prop={'size': 40})


if __name__ == '__main__':
    mean_over_dt()
    plt.show()
