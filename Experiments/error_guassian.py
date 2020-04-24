from __future__ import print_function
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import visual
from statistics import StochasticViewer


class CGPViewer(StochasticViewer):
    """ """
    def __init__(self, file_type='valid', recall_bar=0.5):
        super(CGPViewer, self).__init__(file_type, recall_bar, nrows=1, ncols=1)
        self.fig1, self.ax1 = plt.subplots(nrows=1, ncols=1)
        self.fig2, self.ax2 = plt.subplots(nrows=1, ncols=1)
        self.fontsize = 70
        self.ax1.tick_params(labelsize=self.fontsize)
        self.ax1.set_xlabel('error[m]', fontsize=self.fontsize)
        self.ax1.set_ylabel('probability[-]', fontsize=self.fontsize)
        self.ax2.tick_params(labelsize=self.fontsize)

    def view(self, first, last, remain=-1):
        errors, pps = [], []
        for folder in self.pred_folders[first: last]:
            pred_filepath = self.pred_filepath_root + os.sep + folder
            print("Processing " + pred_filepath)
            files = glob.glob(pred_filepath + os.sep + '*.txt')
            rights, op_bias, om_bias = 0., 0., 0.
            x_error, y_error, t_error = [], [], []
            for i, f in enumerate(files):
                # number of the example
                no = f.split('/')[-1].split('_')[0]
                # ground true
                true_file = self.true_filepath_root + os.sep + str(no) + '_way.txt'
                true = np.loadtxt(true_file, delimiter=',')
                # grip the number of points
                tn_obj = true.shape[0] - 2
                # grip the points
                t_prt = true[1:-1, :-1]
                # prediction
                pred = np.loadtxt(f, delimiter=',')
                p_obj = pred[:, -1]
                # grip the number of predicted points
                pn_obj = sum(p_obj > self.recall_bar)
                # grip the predicted points
                p_prt = []
                for row in pred:
                    if row[-1] > self.recall_bar:
                        p_prt.append(row[:-1])
                # check if the numbers of points are equal
                rights += pn_obj == tn_obj
                op_bias += (pn_obj - 1) == tn_obj
                om_bias += (pn_obj + 1) == tn_obj
                # calculate the error of points
                self.errors_when_obj_numbers_equal(
                    t_prt, np.array(p_prt), [x_error, y_error, t_error])
            error = y_error
            res = self.plot_cdf(error, label=r'$\Theta$-error', color='r')
            m, v = res[-1][1], res[-1][0]
            x = m + v * np.random.randn(100000)
            self.plot_cdf(x, pp=False, label=r'f-$\Theta$-error', color='b')
            self.ax1.legend(bbox_to_anchor=(0.45, 0.89), prop={'size': 60})
            # self.ax2.legend(bbox_to_anchor=(0.45, 0.99), prop={'size': 60})
        return errors, pps

    def plot_cdf(self, y, label, linewidth=10, pp=True, color='r'):
        l1 = self.ax1.hist(
            y, 100, normed=1, histtype='step', facecolor='pink',
            alpha=1.0, cumulative=True, rwidth=0.8,
            linewidth=linewidth, label=label, color=color)
        if pp:
            plt.figure()
            res = stats.probplot(y, plot=self.ax2)
            self.ax2.set_xlabel('theoretical quantiles[-]', fontsize=self.fontsize)
            self.ax2.set_ylabel('ordered values [-]', fontsize=self.fontsize)
            self.ax2.set_title('')
            k, b = res[-1][0], res[-1][1]
            self.ax2.scatter(res[0][0], res[0][1], s=500, c='b',
                             zorder=100, label='x-error')
            self.ax2.plot([-3.5, 3.5], [-3.5*k+b, k*3.5+b], linewidth=12,
                          zorder=1000, color='r', label='f-x-error')
            return res


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    viewer = CGPViewer('valid', recall_bar=0.8)
    viewer.view(2, 3, remain=-1)
    plt.show()
    # 0 vgg19_tiny_free250_check600 - 0.6
    # 1 vgg19_comp_free100_check200 - 0.7
    # 2 vgg19_comp_free200_check300 - 0.7 *** / 0.8 *****
    # 3 vgg19_comp_free200_check400 - 0.7 ** / 0.8 ****
    # 4 vgg19_tiny_free250_check800 - 0.6 *
    # 5 vgg19_comp_free100_check300 - 0.7

