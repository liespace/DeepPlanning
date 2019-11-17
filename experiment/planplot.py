from __future__ import print_function
import os
import matplotlib.pyplot as plt
import visual
import cv2
from search import SearchEngine
import visualization as vsl
import numpy as np

class PlanPlot(SearchEngine):
    """ """
    def __init__(self, recall_bar, file_type, error_mask=(0., 0., 0.),
                 target_number=1000, diff=0):
        SearchEngine.__init__(self, recall_bar, file_type, error_mask,
                              target_number, diff)

    def plot_responses(self, res):
        number = len(res[0])
        for i in range(number):
            no, true, p_path = res[1][i], res[2][i], res[3][i]
            self.plot_true_pred(no, true, p_path)

    def plot_true_pred(self, no, true, p_path):
        # grid map
        grid_file = self.grid_filepath_root + os.sep + str(
            no) + '_gridmap.png'
        grid = cv2.imread(filename=grid_file, flags=-1)
        true_file = self.true_filepath_root + os.sep + str(no) + '_way.txt'
        true = np.loadtxt(true_file, delimiter=',')
        # show
        plt.figure(figsize=(10, 10))
        plt.clf()
        plt.axis('off')
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        poster = plt
        vsl.plot_grid(
            grid, res=0.1, wic=600, hic=600,
            dot_size=10., marker='s', zorder=0,
            poster=poster)
        vsl.plot_true(
            true, curve=True, keys=False, car=True,
            curve_zorder=100, key_zorder=20, car_zorder=200,
            step_size=0.5, rho=4.8, width=4.6, height=1.8,
            car_color='b', car_fill=False, car_width=2, car_alpha=1.0,
            key_color='b', key_width=1, key_fill=False, key_alpha=1.0,
            curve_color='b', curve_style='-', curve_width=4,
            poster=poster)
        vsl.plot_path(
            p_path, curve=True, keys=False, car=True,
            curve_zorder=400, key_zorder=200, car_zorder=400,
            step_size=0.5, rho=4.8, width=4.6, height=1.8,
            car_color='r', car_fill=False, car_width=2, car_alpha=1.0,
            key_color='g', key_width=1, key_fill=False, key_alpha=1.0,
            curve_color='r', curve_style='-', curve_width=4,
            poster=poster)
        vsl.plot_cond(
            true, colors=('#48C9B0', 'g'), width=4.6, height=1.8,
            zorder=1, linewidth=1, fill=True, alpha=1.0,
            poster=poster)
        print('seeing Number: ' + str(no))
        plt.show()

if __name__ == '__main__':
    viewer = PlanPlot(recall_bar=0.8, file_type='valid')
    fs, fd = viewer.find_files(2)

    # viewer.target_diff = 0
    # response = viewer.find_object(files=fs, fun=viewer.check_predicted_number_of_obj)
    # print('ALL Right Obj Prediction Num: %d' % len(response[0]))
    # viewer.error_mask = (4.5, -1, -1.0)  # [2.128*2, 2.092*2, 0.464*2]
    # response = viewer.find_object(files=response[0], fun=viewer.check_prediction_error)
    # print('Error Num: %d' % len(response[0]))
    # response = viewer.find_object(files=response[0], fun=viewer.check_collision)
    # print('Collision-Free Num: %d' % len(response[0]))
    # print(response[1])
    for p in [8263, 370, 3990, 6186, 3770, 3625, 4810, 5220]:  #
        viewer.target_number = p
        response = viewer.find_object(files=fs, fun=viewer.check_number)
        # response = viewer.find_object(files=response[0], fun=viewer.check_collision)
        viewer.plot_responses(response)
        # print(response[0])
        # plt.show()

    # 0 vgg19_tiny_free250_check600 - 0.6
    # 1 vgg19_comp_free100_check200 - 0.7
    # 2 vgg19_comp_free200_check300 - 0.7 **** / 0.8 *****
    # 3 vgg19_comp_free200_check400 - 0.7 ** / 0.8 ***
    # 4 vgg19_tiny_free250_check800 - 0.6 *
    # 5 vgg19_comp_free100_check300 - 0.7
