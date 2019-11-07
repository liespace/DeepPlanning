from __future__ import print_function
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import visual
import cv2


class SimilarityViewer:
    """ """
    def __init__(self, recall_bar, file_type,
                 error_mask=(0., 0., 0.), target_number=1000, diff=0):
        self.recall_bar = recall_bar
        self.pred_filepath_root = 'pred/' + file_type
        self.true_filepath_root = 'dataset' + os.sep + 'well'
        self.grid_filepath_root = 'dataset' + os.sep + 'blue'
        self.pred_folders = os.listdir(self.pred_filepath_root)
        self.fig, self.axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

        self.error_mask = error_mask
        self.target_number = target_number
        self.target_diff = diff

    def find_files(self, number):
        for folder in self.pred_folders[number: number+1]:
            pred_filepath = self.pred_filepath_root + os.sep + folder
            print("Processing " + pred_filepath)
            files = glob.glob(pred_filepath + os.sep + '*.txt')
            return files, folder

    def check_collision(self, true, p_path, no):
        # grid map
        grid_file = self.grid_filepath_root + os.sep + str(
            no) + '_gridmap.png'
        grid = cv2.imread(filename=grid_file, flags=-1)
        return True

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
        visual.plot_grid(grid)
        visual.plot_way(true, 0.5, rho=5., car_size=1.0,
                        car_color='g', line_color='g', point_color='g')
        visual.plot_way(p_path, 0.5, rho=5.,
                        car_color='r', line_color='r', point_color='r')

    def find_object(self, files, fun):
        res = [[], [], [], []]
        for i, f in enumerate(files):
            # number of the example
            no = int(f.split('/')[-1].split('_')[0])
            # ground true
            true_file = self.true_filepath_root + os.sep + str(no) + '_way.txt'
            true = np.loadtxt(true_file, delimiter=',')
            true = true[:, :-1]
            # grip the predicted points
            pred = np.loadtxt(f, delimiter=',')
            p_prt = []
            for row in pred:
                if row[-1] > self.recall_bar:
                    p_prt.append(row[:-1])
            # form the predicted path
            p_path = p_prt[:]
            p_path.insert(0, true[0])
            p_path.append(true[-1])
            p_path = np.array(p_path)

            # checking
            if fun(no, true, p_path):
                res[0].append(f)
                res[1].append(no)
                res[2].append(true)
                res[3].append(p_path)
        return res

    def check_predicted_number_of_obj(self, no, true, p_path):
        # check if the numbers of points are equal
        return (p_path.shape[0] - true.shape[0]) == self.target_diff

    def check_prediction_error(self, no, true, p_path):
        if true.shape[0] > 2:
            q = true[1:-1, :] - p_path[1:-1, :]
            q = np.abs(q)
            x_e, y_e, t_e = max(q[:, 0]), max(q[:, 1]), max(q[:, 2])
            print('x_errors: {}, y_errors:{}, t_errors: {}'.
                  format(x_e, y_e, t_e))
            ind = np.array([x_e, y_e, t_e]) > np.array(self.error_mask)
            return np.all(ind)
        return False

    def check_number(self, no, true, p_path):
        return no == self.target_number

    @staticmethod
    def check_sequence(no, true, p_path):
        return not no % 2 == -1


if __name__ == '__main__':
    viewer = SimilarityViewer(recall_bar=0.7, file_type='valid')
    fs, fd = viewer.find_files(3)

    viewer.target_number = 385
    response = viewer.find_object(files=fs, fun=viewer.check_collision)
    print(len(response[0]))

    # viewer.plot_responses(response)
    # plt.show()
