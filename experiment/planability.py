from __future__ import print_function
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import visual
import csv


class PlanChecker(object):
    """ """
    def __init__(self, file_type='valid', base_type='rrt', obj_type='dwa-rrt'):
        work_dir = 'experiment' + os.sep + 'backup'
        self.base_filepath = work_dir + os.sep + file_type + os.sep + base_type
        self.obj_filepath = work_dir + os.sep + file_type + os.sep + obj_type

    def check_obj(self, save=False):
        # obj
        obj_files = self.find_files(self.obj_filepath)
        obj_res = self.resolve_infos(obj_files, self.obj_filepath)
        obj_sfs = self.resolve_res(obj_res)
        self.print_sf(obj_sfs, len(obj_files), self.obj_filepath)
        if save:
            self.save_sf(obj_sfs, len(obj_files), self.obj_filepath)

    def check_base(self, save=False):
        # base
        base_files = self.find_files(self.base_filepath)
        base_res = self.resolve_infos(base_files, self.base_filepath)
        base_sfs = self.resolve_res(base_res)
        # print or save
        self.print_sf(base_sfs, len(base_files), self.base_filepath)
        if save:
            self.save_sf(base_sfs, len(base_files), self.base_filepath)

    def diff(self, save=False):
        # obj
        obj_nos = self.find_files(self.obj_filepath)
        obj_res = self.resolve_infos(obj_nos, self.obj_filepath)
        # base
        base_res = self.resolve_infos(obj_nos, self.base_filepath)
        # diff
        diff_res = []
        for i in range(len(obj_res)):
            dr = np.array(obj_res[i]) - np.array(base_res[i])
            diff_res.append(dr)
        diff_sf = self.resolve_res(diff_res)
        self.print_sf(diff_sf, len(obj_nos), self.obj_filepath)
        if save:
            self.save_sf(diff_sf, len(obj_nos), self.obj_filepath, 'diff_se')
        channel = 1
        arr = np.array(diff_res)[channel, :]
        nos = arr.argsort()[:10]
        # no = np.argmin(arr[channel, :])
        print([self.extract_no(obj_nos[n]) for n in nos], [diff_res[channel][n] for n in nos])

    @staticmethod
    def print_sf(sfs, amount, filepath='base'):
        print('{} Statistical Features are:'.format(filepath))
        print('Precision: %.4f%% / %d' % (amount / 1725. * 100, 1725))
        print('ST:  %.4f / %.4f' % (sfs[0][0], sfs[0][1]))
        print('TFF-M: %.4f / %.4f' % (sfs[1][0], sfs[1][1]))
        print('TFF-S: %.4f / %.4f' % (sfs[2][0], sfs[2][1]))
        print('Cost-M: %.4f / %.4f' % (sfs[3][0], sfs[3][1]))
        print('Cost-S: %.4f / %.4f' % (sfs[4][0], sfs[4][1]))
        print('Cost2-M: %.4f / %.4f' % (sfs[5][0], sfs[5][1]))
        print('Cost2-S: %.4f / %.4f' % (sfs[6][0], sfs[6][1]))

    @staticmethod
    def resolve_res(res):
        sts_sf = [np.mean(res[0]), np.sqrt(np.var(res[0]))]
        tff_ms_sf = [np.mean(res[1]), np.sqrt(np.var(res[1]))]
        tff_ss_sf = [np.mean(res[2]), np.sqrt(np.var(res[2]))]
        cost_ms_sf = [np.mean(res[3]), np.sqrt(np.var(res[3]))]
        cost_ss_sf = [np.mean(res[4]), np.sqrt(np.var(res[4]))]
        cost2_ms_sf = [np.mean(res[5]), np.sqrt(np.var(res[5]))]
        cost2_ss_sf = [np.mean(res[6]), np.sqrt(np.var(res[6]))]
        return (sts_sf, tff_ms_sf, tff_ss_sf,
                cost_ms_sf, cost_ss_sf, cost2_ms_sf, cost2_ss_sf)

    @staticmethod
    def save_sf(sfs, amount, filepath, filename='statistical_features'):
        res_file = filepath + os.sep + filename + '.csv'
        with open(res_file, mode='w') as csv_file:
            fieldnames = [
                'SR', 'ST-M', 'ST-SE',
                'TFF-M-Mean', 'TFF-M-SE', 'TFF-S-Mean', 'TFF-S-SE',
                'Cost-M-Mean', 'Cost-M-SE', 'Cost-S-Mean', 'Cost-S-SE',
                'Cost2-M-Mean', 'Cost2-M-SE', 'Cost2-S-Mean', 'Cost2-S-SE']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                'SR': amount / 1725. * 100,
                'ST-M': sfs[0][0], 'ST-SE': sfs[0][1],
                'TFF-M-Mean': sfs[1][0], 'TFF-M-SE': sfs[1][1],
                'TFF-S-Mean': sfs[2][0], 'TFF-S-SE': sfs[2][1],
                'Cost-M-Mean': sfs[3][0], 'Cost-M-SE': sfs[3][1],
                'Cost-S-Mean': sfs[4][0], 'Cost-S-SE': sfs[4][1],
                'Cost2-M-Mean': sfs[5][0], 'Cost2-M-SE': sfs[5][1],
                'Cost2-S-Mean': sfs[6][0], 'Cost2-S-SE': sfs[6][1]})

    def resolve_infos(self, files, filepath):
        tff_ms, tff_ss, sts = [], [], []
        cost_ms, cost_ss, cost2_ms, cost2_ss = [], [], [], []
        for f in files:
            no = self.extract_no(f)
            st, tff, cost, cost2 = self.retrieve_infos(no, filepath)
            tff_m, tff_s = np.mean(tff), np.sqrt(np.var(tff))
            cost_m, cost_s = np.mean(cost), np.sqrt(np.var(cost))
            cost2_m, cost2_s = np.mean(cost2), np.sqrt(np.var(cost2))
            # print(tff_m, tff_s, cost_m, cost_s, cost2_m, cost2_s)
            sts.append(st)
            tff_ms.append(tff_m)
            tff_ss.append(tff_s)
            cost_ms.append(cost_m)
            cost_ss.append(cost_s)
            cost2_ms.append(cost2_m)
            cost2_ss.append(cost2_s)
        return sts, tff_ms, tff_ss, cost_ms, cost_ss, cost2_ms, cost2_ss

    @staticmethod
    def retrieve_infos(no, filepath):
        st_file = filepath + os.sep + str(no) + '_st.txt'
        tff_file = filepath + os.sep + str(no) + '_tff.txt'
        cost_file = filepath + os.sep + str(no) + '_cost.txt'
        cost2_file = filepath + os.sep + str(no) + '_cost2.txt'
        st = np.loadtxt(st_file, delimiter=',')
        tff = np.loadtxt(tff_file, delimiter=',')
        cost = np.loadtxt(cost_file, delimiter=',')
        cost2 = np.loadtxt(cost2_file, delimiter=',')
        return st, tff, cost, cost2


    @staticmethod
    def extract_no(filename):
        return int(filename.split('/')[-1].split('_')[0])

    @staticmethod
    def find_files(filepath, form='*_tff.txt'):
        return glob.glob(filepath + os.sep + form)

if __name__ == '__main__':
    base = 'rrt'
    obj = 'dwb-rrt-l' + os.sep + 'vgg19_comp_free200_check300_0.8'
    # obj = 'rrt-fast'
    checker = PlanChecker(file_type='valid', base_type=base, obj_type=obj)
    # checker.check_obj(True)
    checker.diff(False)

# vgg19_comp_free200_check400_0.7
