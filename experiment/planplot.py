from __future__ import print_function
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import visual
import cv2
from rrt.planner import Planner
from rrt.dtype import State, Location, Rotation, Velocity, C2GoType
import reeds_shepp
from scipy import stats
import similaritymeasures
from search import SearchEngine
import time


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
        # show
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        visual.plot_grid(grid)
        visual.plot_way(true, 0.5, rho=5., car_size=1.0,
                        car_color='g', line_color='g', point_color='g')
        visual.plot_way(p_path, 0.5, rho=5.,
                        car_color='r', line_color='r', point_color='r')


if __name__ == '__main__':
    viewer = SearchEngine(recall_bar=0.8, file_type='valid')
    fs, fd = viewer.find_files(2)

    # viewer.target_diff = 0
    # response = viewer.find_object(files=fs, fun=viewer.check_predicted_number_of_obj)
    # print('ALL Right Obj Prediction Num: %d' % len(response[0]))
    # viewer.error_mask = (-1, 4.0, 1.0)  # [2.128*2, 2.092*2, 0.464*2]
    # response = viewer.find_object(files=response[0], fun=viewer.check_prediction_error)
    # print('Error Num: %d' % len(response[0]))
    # response = viewer.find_object(files=response[0], fun=viewer.check_collision)
    # print('Collision-Free Num: %d' % len(response[0]))
    # print(response[1])

    viewer.target_number = 8600
    response = viewer.find_object(files=fs, fun=viewer.check_number)
    response = viewer.find_object(files=response[0], fun=viewer.check_collision)
    viewer.plot_responses(response)
    # print(response[0])
    plt.show()

    # 0 vgg19_tiny_free250_check600 - 0.6
    # 1 vgg19_comp_free100_check200 - 0.7
    # 2 vgg19_comp_free200_check300 - 0.7 **** / 0.8 *****
    # 3 vgg19_comp_free200_check400 - 0.7 ** / 0.8 ***
    # 4 vgg19_tiny_free250_check800 - 0.6 *
    # 5 vgg19_comp_free100_check300 - 0.7
