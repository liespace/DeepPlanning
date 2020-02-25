from __future__ import print_function
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import visual
import csv


class DataViewer(object):
    """ """
    def __init__(self, file_type='free_epoch'):
        plt.rcParams["font.family"] = "Times New Roman"
        work_dir = 'experiment' + os.sep + 'train'
        self.filepath = work_dir + os.sep + file_type
        self.fontsize = 40
        self.key = 'cor'
        self.aux = 'obj'
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_ylabel('C[-]', fontsize=self.fontsize)
        self.ax1.set_xlabel('epoch[-]', fontsize=self.fontsize)
        self.ax2 = self.ax1.twinx()  # this is the important function
        self.ax2.set_ylabel('N[-]', fontsize=self.fontsize)
        self.ax1.tick_params(labelsize=self.fontsize)
        self.ax2.tick_params(labelsize=self.fontsize)
        self.lines = (
            [1, 0],  # solid line
            [6, 2],  # dotted line
            [4, 2, 1, 2],  # long dotted line
            [4, 2, 1, 1, 1, 2],  # line-dot
            [3, 2],  # dots
            [1, 1])

    def view(self, begin=0, end=-1, window=30, prefixes=('', '', '', ''),
             anchor=(0.98, 0.78), colors=('b', 'g', 'magenta'),
             line_styles=(('-', '--'), ('-', '--'), ('-', '--')),
             yt1=np.arange(0.042, 0.050, 0.002),
             yt2=np.arange(0.930, 0.9425, 0.0035), zorders=(50, 50, 50)):
        files = self.find_files(self.filepath)
        handles, labels = [], []
        for i, f in enumerate(files):
            pf = self.find_partner(f)
            key = self.read_csv(f)
            aux = self.read_csv(pf)
            name = f.split('free_')[-1].split('-')[0]
            hs, ls = self.plot_free_epoch(
                key, aux, name, zorder=zorders[i],
                window=window, yt1=yt1, yt2=yt2, prefix=prefixes[i],
                begin=begin, end=end, color=colors[i], dashes=line_styles[i])
            handles.extend(hs)
            labels.extend(ls)
        plt.legend(handles, labels, bbox_to_anchor=anchor, prop={'size': 38})
        plt.show()
        return

    def find_partner(self, filename):
        parts = filename.split(self.key)
        partner = parts[0] + self.aux + parts[-1]
        return partner

    def plot_free_epoch(self, key, aux, name, window=20, prefix='', zorder=50,
                        begin=0, end=-1, color='g', dashes=('-', '--'),
                        yt1=np.arange(0.042, 0.050, 0.002),
                        yt2=np.arange(0.930, 0.9425, 0.0035)):
        self.ax1.set_yticks(yt1)
        self.ax2.set_yticks(yt2)
        x = key[begin:end, 0]
        dk = pd.DataFrame(key[begin:end, 1])
        da = pd.DataFrame(aux[begin:end, 1])
        dk = dk[0].rolling(window).mean()
        da = da[0].rolling(window).mean()
        l1, = self.ax1.plot(
            x, dk, dashes=dashes[0], linewidth=6, color=color, zorder=zorder)
        l2, = self.ax2.plot(
            x, da, dashes=dashes[1], linewidth=6, color=color, zorder=zorder)
        b1 = prefix + '-' + 'C'
        b2 = prefix + '-' + 'N'
        # draw vline
        no, = np.where(key[:, 0] == int(name))
        self.ax1.scatter(int(name), dk[no-begin],
                         marker='h', s=600, zorder=zorder, c=color)
        self.ax2.scatter(int(name), da[no-begin],
                         marker='h', s=600, zorder=zorder, c=color)
        return [l1, l2], [b1, b2]

    @staticmethod
    def read_csv(filename):
        values = []
        with open(filename, 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                values.append([int(row['Step']), float(row['Value'])])
        return np.array(values)

    @staticmethod
    def find_files(filepath, form='*_cor_metric.csv'):
        return glob.glob(filepath + os.sep + form)


if __name__ == '__main__':
    # viewer = DataViewer(file_type='free_epoch')
    # styles = (
    #     [viewer.lines[0], viewer.lines[1]],
    #     [viewer.lines[2], viewer.lines[3]],
    #     [viewer.lines[4], viewer.lines[5]])
    # prefixes = ['fe-200', 'fe-150', 'fe-100']
    # zs = [100, 0, 200]
    # viewer.view(begin=30, end=350, window=30, anchor=(0.98, 0.78),
    #             colors=('r', 'g', 'b'), line_styles=styles,
    #             prefixes=prefixes, zorders=zs)

    viewer = DataViewer(file_type='diff_free')
    styles = (
        [viewer.lines[0], viewer.lines[1]],
        [viewer.lines[4], viewer.lines[5]],
        [viewer.lines[2], viewer.lines[3]])
    prefixes = ['YIPS-V-8', 'YIPS-V-7', 'YIPS-V-6']
    zs = [200, 100, 0]
    viewer.view(begin=30, end=350, window=20, anchor=(0.95, 0.75),
                colors=('r', 'g', 'b'), line_styles=styles, prefixes=prefixes,
                yt1=np.arange(0.042, 0.080, 0.005),
                yt2=np.arange(0.850, 0.9425, 0.01),
                zorders=[100, 0, 200])

# vgg19_comp_free200_check400_0.7
