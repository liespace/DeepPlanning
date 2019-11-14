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
c23_np = np.array([1140, 1180, 1195, 1217, 1229, 1242, 1253, 1264, 1257])
c12_np = np.array([935, 1057, 1146, 1204, 1234, 1247, 1255, 1266, 1244])
t28_np = np.array([1041, 1133, 1184, 1209, 1231, 1245, 1247, 1233, 1218])
plt.rcParams["font.family"] = "Times New Roman"

def acc_over_dt():
    # accuracy over threshold
    fontsize = 50
    linewidth = 8
    color = ['b', 'r', 'g']
    linestyle = ['--', '-', ':']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('accuracy[-]', fontsize=fontsize)
    ax1.set_xlabel('discrimination threshold[-]', fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax1.set_xticks(np.arange(0, 1, 0.1))
    # ax1.set_yticks(yt1)
    l1, = ax1.plot(threshold, c12_np / amount,
                   linewidth=linewidth,
                   color=color[0], zorder=50,
                   linestyle=linestyle[0])
    l2, = ax1.plot(threshold, c23_np / amount,
                   linewidth=linewidth,
                   color=color[1], zorder=100,
                   linestyle=linestyle[1])
    l3, = ax1.plot(threshold, t28_np / amount,
                   linewidth=linewidth,
                   color=color[2], zorder=0,
                   linestyle=linestyle[2])
    ax1.scatter(threshold, c12_np/ amount, marker='D',
                s=500, c=color[0], zorder=50)
    ax1.scatter(threshold, c23_np/ amount, marker='X',
                s=1000, c=color[1], zorder=100)
    ax1.scatter(threshold, t28_np/ amount, marker='^',
                s=500, c=color[2], zorder=0)

    plt.legend([l1, l2, l3], ['YIPS-V-6', 'YIPS-V-8', 'YIPS-V-VT'],
               bbox_to_anchor=(0.9, 0.7), prop={'size': 40})


if __name__ == '__main__':
    acc_over_dt()
    plt.show()
