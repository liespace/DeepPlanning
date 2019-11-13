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
    def __init__(self, file_type='free_step'):
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

    def read_data(self):
        files = self.find_files(self.filepath)
        handles, labels = [], []
        for f in files:
            pf = self.find_partner(f)
            key = self.read_csv(f)
            aux = self.read_csv(pf)
            name = f.split('free_')[-1].split('-')[0]
            hs, ls = self.plot_free_data(key, aux, name, window=30, begin=40, end=350)
            handles.extend(hs)
            labels.extend(ls)
        plt.legend(handles, labels, bbox_to_anchor=(0.98, 0.78), prop={'size': 38})
        plt.show()
        return

    def find_partner(self, filename):
        parts = filename.split(self.key)
        partner = parts[0] + self.aux + parts[-1]
        return partner

    def plot_free_data(self, key, aux, name, window=20, begin=0, end=-1):
        self.ax1.set_yticks(np.arange(0.042, 0.050, 0.002))
        self.ax2.set_yticks(np.arange(0.930, 0.9425, 0.0035))
        dk = pd.DataFrame(key[begin:end, 1])
        da = pd.DataFrame(aux[begin:end, 1])
        dk = dk[0].rolling(window).mean()
        da = da[0].rolling(window).mean()
        l1, = self.ax1.plot(dk, linewidth=6)
        l2, = self.ax2.plot(da, linestyle='--', linewidth=6)
        b1 = 'fe' + name + '-' + 'C'
        b2 = 'fe' + name + '-' + 'N'
        # draw vline
        self.ax1.scatter(int(name), dk[int(name)], marker='h', s=600, zorder=50)
        self.ax2.scatter(int(name), da[int(name)], marker='h', s=600, zorder=50)
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
    viewer = DataViewer(file_type='free_step')
    viewer.read_data()

# vgg19_comp_free200_check400_0.7
