from __future__ import print_function
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import csv


class DataPlotter(object):
    """ """

    def __init__(self, file_type='free_epoch'):
        plt.rcParams["font.family"] = "Times New Roman"
        self.filepath = 'logs'
        self.fontsize = 55
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

    def plot_obj_loss_val(self):
        files = self.find_files(self.filepath, form='*epoch_val_dw_obj_loss.csv')
        files.sort()
        colors = ['green', 'red', 'blue']
        ax, handles, labels = self.new_figure(y_label='$J_c$[-]'), [], ['Normal', 'Fine-tuning']
        for i, f in enumerate(files):
            data2d, lamb = self.read_csv(f), float(f.split('bce_')[-1].split('_')[0])
            handles.append(self.plotting(ax=ax, data2d=data2d, end=300, color=colors[i]))
        # ax.set_ylim([0.1, 0.2])
        # ax.set_yticks(np.arange(0.1, 0.2, 0.1))
        plt.legend(handles, labels, prop={'size': self.fontsize}, ncol=1)

    def plot_obj_loss(self):
        files = self.find_files(self.filepath, form='*epoch_dw_obj_loss.csv')
        files.sort()
        colors = ['green', 'red', 'blue']
        ax, handles, labels = self.new_figure(y_label='$J_c$[-]'), [], []
        for i, f in enumerate(files):
            data2d, lamb = self.read_csv(f), float(f.split('bce_')[-1].split('_')[0])
            handles.append(self.plotting(ax=ax, data2d=data2d, end=250, color=colors[i]))
            lr = float(f.split('adam_')[-1].split(')')[0])
            labels.append('$lr$ ={:1.0e}'.format(lr))
        ax.set_ylim([0.0, 0.4])
        ax.set_yticks(np.arange(0.0, 0.4, 0.1))
        plt.legend(handles, labels, prop={'size': self.fontsize}, ncol=1)

    def plot_cor_loss_val(self):
        files = self.find_files(self.filepath, form='*epoch_val_dw_cor_loss.csv')
        files.sort()
        colors = ['green', 'red', 'blue']
        ax, handles, labels = self.new_figure(), [], ['Normal', 'Fine-tuning']
        for i, f in enumerate(files):
            data2d, lamb = self.read_csv(f), float(f.split('bce_')[-1].split('_')[0])
            data2d[:, 1] /= float(lamb)
            handles.append(self.plotting(ax=ax, data2d=data2d, end=300, color=colors[i]))
        plt.legend(handles, labels, prop={'size': self.fontsize}, ncol=1)

    def plot_cor_loss(self):
        files = self.find_files(self.filepath, form='*epoch_dw_cor_loss.csv')
        files.sort()
        colors = ['green', 'red', 'blue']
        ax, handles, labels = self.new_figure(), [], []
        for i, f in enumerate(files):
            data2d, lamb = self.read_csv(f), float(f.split('bce_')[-1].split('_')[0])
            data2d[:, 1] /= float(lamb)
            handles.append(self.plotting(ax=ax, data2d=data2d, end=200, color=colors[i]))
            lr = float(f.split('adam_')[-1].split(')')[0])
            labels.append('$lr$ ={:1.0e}'.format(lr))
        # ax.set_ylim([0.0, 0.])
        ax.set_yticks(np.arange(0.22, 0.255, 0.01))
        plt.legend(handles, labels, prop={'size': self.fontsize}, ncol=1)

    def plotting(self, ax, data2d, begin=0, end=-1, smoothing=0.8, color=None):
        x = data2d[begin:end, 0]
        y = data2d[begin:end, 1]
        # y = pd.DataFrame(data2d[begin:end, 1])
        l1, = ax.plot(x, self.smooth(y, smoothing), linewidth=8, color=color)
        return l1

    @staticmethod
    def smooth(scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value
        return smoothed

    def new_figure(self, y_label='$J_r/\\lambda_r$[-]', x_label='hour[-]'):
        fig = plt.figure()
        ax = fig.gca()
        ax.set_ylabel(y_label, fontsize=self.fontsize)
        ax.set_xlabel(x_label, fontsize=self.fontsize)
        ax.tick_params(labelsize=self.fontsize)
        return ax

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
        self.ax1.scatter(int(name), dk[no - begin],
                         marker='h', s=600, zorder=zorder, c=color)
        self.ax2.scatter(int(name), da[no - begin],
                         marker='h', s=600, zorder=zorder, c=color)
        return [l1, l2], [b1, b2]

    @staticmethod
    def read_csv(filename):
        values = []
        with open(filename, 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                values.append([float(row['Wall time']), float(row['Value'])])
        values = np.array(values)
        values[:, 0] -= values[0, 0]
        values[:, 0] /= 3600
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

    viewer = DataPlotter(file_type='diff_free')
    styles = (
        [viewer.lines[0], viewer.lines[1]],
        [viewer.lines[4], viewer.lines[5]],
        [viewer.lines[2], viewer.lines[3]])
    prefixes = ['YIPS-V-8', 'YIPS-V-7', 'YIPS-V-6']
    zs = [200, 100, 0]
    # viewer.view(begin=30, end=350, window=20, anchor=(0.95, 0.75),
    #             colors=('r', 'g', 'b'), line_styles=styles, prefixes=prefixes,
    #             yt1=np.arange(0.042, 0.080, 0.005),
    #             yt2=np.arange(0.850, 0.9425, 0.01),
    #             zorders=[100, 0, 200])

    viewer.plot_cor_loss_val()
    viewer.plot_obj_loss_val()
    # viewer.plot_cor_loss()
    # viewer.plot_obj_loss()
    plt.show()

# vgg19_comp_free200_check400_0.7
