from __future__ import print_function
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import visual


class StochasticViewer(object):
    """ """
    def __init__(self, file_type='valid', recall_bar=0.5, nrows=3, ncols=3):
        self.recall_bar = recall_bar
        self.pred_filepath_root = 'pred/' + file_type
        self.true_filepath_root = 'dataset' + os.sep + 'well'
        self.pred_folders = os.listdir(self.pred_filepath_root)
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols)

    def view(self, first, last, remain=-1):
        errors, pps = [], []
        for folder in self.pred_folders[first: last]:
            pred_filepath = self.pred_filepath_root + os.sep + folder
            print("Processing " + pred_filepath)
            files = glob.glob(pred_filepath + os.sep + '*.txt')
            rights, op_bias, om_bias = 0., 0., 0.
            x_error, y_error, t_error = [], [], []
            for i, f in enumerate(files):
                if i % 2 == remain:
                    continue
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
            self.show_info(
                x_error, y_error, t_error, rights, om_bias, op_bias, len(files))
            pps.append(self.plot_pdf_cdf(x_error, y_error, t_error))
            errors.append([x_error, y_error, t_error])
        return errors, pps

    @staticmethod
    def show_info(x_e, y_e, t_e, rights, om_bias, op_bias, number):
        sx, sy, st = pd.Series(x_e), pd.Series(y_e), pd.Series(t_e)
        print('X-mean: %2.6f, X-SE: %2.6f, '
              'X-max: %2.6f, X-min: %2.6f, X-skew: %2.6f, X-kurt: %2.6f' %
              (np.mean(x_e), np.sqrt(np.var(x_e)),
               np.max(x_e), np.min(x_e), sx.skew(), sx.kurt()))
        print('Y-mean: %2.6f, Y-SE: %3.6f, '
              'Y-max: %2.6f, Y-min: %2.6f, Y-skew: %2.6f, Y-kurt: %2.6f' %
              (np.mean(y_e), np.sqrt(np.var(y_e)),
               np.max(y_e), np.min(y_e), sy.skew(), sy.kurt()))
        print('T-mean: %2.6f, T-SE: %2.6f, '
              'T-max: %2.6f, T-min: %2.6f, T-skew: %2.6f, T-kurt: %2.6f' %
              (np.mean(t_e), np.sqrt(np.var(t_e)),
               np.max(t_e), np.min(t_e), st.skew(), st.kurt()))
        # stochastic performance
        precision = rights / number * 100
        ob_precision = (rights + op_bias + om_bias) / number * 100
        print('Precision: %2.4f%%, OB-Precision: %2.4f%%, '
              'Rights: %5d, OP-B: %5d OM-B: %5d' %
              (precision, ob_precision, rights, op_bias, om_bias))

    @staticmethod
    def errors_when_obj_numbers_equal(t_prt, p_prt, errors):
        if t_prt.shape[0] == p_prt.shape[0] and t_prt.shape[0] > 0:
            q = t_prt - np.array(p_prt)
            errors[0].extend(q[:, 0])
            errors[1].extend(q[:, 1])
            errors[2].extend(q[:, 2])

    @staticmethod
    def errors_without_considering_obj_number(t_prt, p_prt, errors):
        mask = (0.8, 0.2)
        if t_prt.shape[0] > 0:
            for p in p_prt:
                q = t_prt - p
                dxy = np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2)
                dt = np.abs(q[:, -1])
                ds = dxy * mask[0] + dt * mask[1]
                index = np.argmin(ds)
                errors[0].append(q[index][0])
                errors[1].append(q[index][1])
                errors[2].append(q[index][2])

    def plot_pdf_cdf(self, x, y, t, pp=True):
        ps = []
        linewidth = 4
        # x error
        self.axes[0, 0].set_title('pdf-x')
        self.axes[0, 0].hist(
            x, 100, weights=np.ones_like(x) / float(len(x)),
            histtype='step', facecolor='yellowgreen', linewidth=linewidth)
        self.axes[1, 0].set_title(r"CDF of error on $x$")
        self.axes[1, 0].hist(
            x, 100, normed=1, histtype='step', facecolor='pink',
            alpha=0.75, cumulative=True, rwidth=0.8, linewidth=linewidth)
        if pp:
            self.axes[2, 0].set_title("p-p-x")
            res = stats.probplot(x, plot=self.axes[2, 0])
            ps.append(res[-1])
        # y error
        self.axes[0, 1].set_title('pdf-y')
        self.axes[0, 1].hist(
            y, 100, weights=np.ones_like(y) / float(len(y)),
            histtype='step', facecolor='yellowgreen', linewidth=linewidth)
        self.axes[1, 1].set_title(r"CDF of error on $y$")
        self.axes[1, 1].hist(
            y, 100, normed=1, histtype='step', facecolor='pink',
            alpha=0.75, cumulative=True, rwidth=0.8, linewidth=linewidth)
        if pp:
            self.axes[2, 1].set_title("p-p-y")
            res = stats.probplot(y, plot=self.axes[2, 1])
            ps.append(res[-1])
        # theta error
        self.axes[0, 2].set_title('pdf-theta')
        self.axes[0, 2].hist(
            t, 100, weights=np.ones_like(t) / float(len(t)),
            histtype='step', facecolor='yellowgreen', linewidth=linewidth)
        self.axes[1, 2].set_title(r"CDF of error on $\theta$")
        self.axes[1, 2].hist(
            t, 100, normed=1, histtype='step', facecolor='pink',
            alpha=0.75, cumulative=True, rwidth=0.8, linewidth=linewidth)
        self.fig.subplots_adjust(hspace=0.4)
        if pp:
            self.axes[2, 2].set_title("p-p-t")
            res = stats.probplot(t, plot=self.axes[2, 2])
            ps.append(res[-1])
        return ps

    def fitted_gaussian_dist(
            self, ps, size=1000000, pp=False, mask=(1., 1., 1.)):
        x_m, x_v = ps[0][0][1], ps[0][0][0] * mask[0]
        y_m, y_v = ps[0][1][1], ps[0][1][0] * mask[1]
        t_m, t_v = ps[0][2][1], ps[0][2][0] * mask[2]
        print('f-X-m: %2.6f, f-X-SE: %2.6f' % (x_m, x_v))
        print('f-Y-m: %2.6f, f-Y-SE: %2.6f' % (y_m, y_v))
        print('f-T-m: %2.6f, f-T-SE: %2.6f' % (t_m, t_v))
        x = x_m + x_v * np.random.randn(size)
        y = y_m + y_v * np.random.randn(size)
        t = t_m + t_v * np.random.randn(size)
        self.plot_pdf_cdf(x, y, t, pp=pp)


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    viewer = StochasticViewer('valid', recall_bar=0.8)
    e_s, p_s = viewer.view(2, 3, remain=-1)
    # plot gaussian pdf anc cdf
    viewer.fitted_gaussian_dist(p_s, mask=(1., 1., 1.))
    plt.show()
    # 0 vgg19_tiny_free250_check600 - 0.6
    # 1 vgg19_comp_free100_check200 - 0.7
    # 2 vgg19_comp_free200_check300 - 0.7 *** / 0.8 *****
    # 3 vgg19_comp_free200_check400 - 0.7 ** / 0.8 ****
    # 4 vgg19_tiny_free250_check800 - 0.6 *
    # 5 vgg19_comp_free100_check300 - 0.7

