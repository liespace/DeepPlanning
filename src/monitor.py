import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.matlib import repmat
import numpy as np
import csv
import os
from PIL import Image


class DataMonitor:
    def __init__(self, dir_parent, menu=None):
        self.menu = menu
        self.dir_parent = dir_parent
        self.fig = plt.figure()
        self.xlim = (-50, 50)
        self.ylim = (-50, 50)
        self.fontsize = 5

    def write_prediction(self, number=None, data=None):
        filepath = '{}/{}prediction.csv'.format(self.dir_parent, number)
        np.savetxt(filepath, data, delimiter=",")
        print('wrote ' + filepath)

    @staticmethod
    def read_csv(file_name, delimiter=','):
        products = csv.reader(open(file_name, newline=''), delimiter=delimiter, quotechar='|')
        my_list = []
        for row in products:
            dozen = []
            for item in row:
                if item is not '':
                    dozen.append(float(item))
            my_list.append(dozen)
        return my_list

    @staticmethod
    def where(benchmark, your_list):
        my_list = []
        for index, item in enumerate(your_list):
            if item > benchmark:
                my_list.append(index)
        return my_list

    def read_input(self, number):
        name_gridmap = '{}/{}gridmap.png'.format(self.dir_parent, number)
        name_cdt = '{}/{}condition.csv'.format(self.dir_parent, number)

        gridmaps = []
        cdts = []
        gridmap = np.array(Image.open(name_gridmap).convert(mode='RGB'), dtype=np.float32)
        gridmap /= 255
        cdt = np.asarray(self.read_csv(name_cdt, delimiter=' '), dtype=np.float32)
        if cdt.shape[0] < 5:
            while cdt.shape[0] < 5:
                cdt = np.append(cdt, [cdt[-1, :]], 0)
        cdt = self.gcs_to_lcs(cdt)[1:, :]
        gridmaps.append(gridmap)
        cdts.append(cdt)
        return np.array(gridmaps), np.array(cdts)

    def gcs_to_lcs(self, condition):
        state = condition[0, 0:2]
        angle = condition[0, 2]
        rotate = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])

        condition_location = condition[:, 0:2]
        condition_yaw = condition[:, 2]

        condition_location = np.dot((condition_location - state), rotate.transpose())
        condition_yaw = (condition_yaw.transpose() - angle) * 10

        lcs_condition = np.c_[condition_location, condition_yaw]

        return lcs_condition

    def show_gridmap(self, number):
        filepath = '{}/{}gridmap.png'.format(self.dir_parent, number)
        grid_map = mpimg.imread(filepath)

        im_size = np.size(grid_map)
        resolution = 0.2
        im_width = grid_map.shape[0]
        im_height = grid_map.shape[1]
        pixels = grid_map.reshape(im_size)

        # Structure the 2d pixels as matrix
        u_coord = repmat(np.r_[0:im_width:1], im_height, 1).reshape(im_size)
        v_coord = repmat(np.c_[0:im_height:1], 1, im_width).reshape(im_size)

        # Find indexes of the obstacle pixels
        selected_indexes = self.where(0, pixels.tolist())

        # Trim coords
        u_coord = np.asarray(list(u_coord[selected_indexes])).transpose()
        v_coord = np.asarray(list(v_coord[selected_indexes])).transpose()

        # Transform the coords
        origins = repmat(np.r_[im_width / 2, im_height / 2], np.size(u_coord), 1).transpose()

        coords = (np.array([u_coord, v_coord]) - origins) * resolution
        plt.scatter(coords[0, :], coords[1, :], s=0.2, marker="s", linewidths=0)

    def show_condition(self, number=0):
        filepath = '{}/{}condition.csv'.format(self.dir_parent, number)
        conditions = np.asarray(self.read_csv(filepath, ' '))
        origin = conditions[0, 0:2]
        theta = -conditions[0][2]
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        rect = plt.Rectangle((-5 / 2, -3 / 2), 5, 3, fill=False, edgecolor='green', linewidth=0.2)
        plt.gca().add_patch(rect)

        path = conditions[1:, 0:2]
        headings = conditions[1:, 2]
        origins = repmat(np.r_[origin], path.shape[0], 1)
        path -= origins
        path = np.dot(rotation_matrix, path.transpose())

        for i in range(path.shape[1]):
            circle = plt.Circle((path[0, i], path[1, i]), radius=1, fill=True)
            plt.gca().add_patch(circle)
            plt.gca().arrow(path[0, i], path[1, i], np.cos(headings[i] + theta), np.sin(headings[i] + theta),
                            head_width=0.5, head_length=1.5, linewidth=0)

    def read_transform(self, number=0):
        filepath = '{}/{}condition.csv'.format(self.dir_parent, number)
        conditions = np.asarray(self.read_csv(filepath, ' '))
        origin = conditions[0, 0:2]
        theta = -conditions[0][2]
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return origin, rotation_matrix, theta

    def show_sproducts(self, number=0):
        filepath = '{}/{}sproducts.csv'.format(self.dir_parent, number)
        my_list = self.read_csv(filepath)
        if not my_list[0]:
            plt.clf()
            return
        origin, rotation_matrix, _ = self.read_transform(number)
        for i in range(int(len(my_list) / 3)):
            coord = np.array([my_list[i * 3], my_list[i * 3 + 1]])
            origins = repmat(np.c_[origin], 1, coord.shape[1])
            coord = np.dot(rotation_matrix, coord - origins)
            plt.scatter(coord[0, :], coord[1, :], s=0.01, marker="x", linewidths=0)

    def show_label(self, number=0):
        filepath = '{}/{}label.csv'.format(self.dir_parent, number)
        origin, rotation_matrix, theta = self.read_transform(number)
        label = np.asarray(self.read_csv(filepath))
        path = label[:, 0:2]
        headings = label[:, 2]
        origins = repmat(np.r_[origin], path.shape[0], 1)
        path -= origins
        path = np.dot(rotation_matrix, path.transpose())

        for i in range(path.shape[1]):
            color = 'g'
            if i == 0:
                color = 'k'
            circle = plt.Circle((path[0, i], path[1, i]), radius=1, fill=False, color=color, linewidth=0.2)
            plt.gca().add_patch(circle)
            plt.gca().arrow(path[0, i], path[1, i], np.cos(headings[i] + theta), np.sin(headings[i] + theta),
                            head_width=0.5, head_length=1.5, linewidth=0, color='g')

    def show_prediction(self, number=0):
        filepath = '{}/{}prediction.csv'.format(self.dir_parent, number)
        prediction = np.asarray(self.read_csv(filepath))
        path = prediction[:, 0:2]
        headings = prediction[:, 2]

        for i in range(path.shape[0]):
            color = 'r'
            circle = plt.Circle((path[i, 0], path[i, 1]), radius=1, fill=False, color=color, linewidth=0.2)
            plt.gca().add_patch(circle)
            plt.gca().arrow(path[i, 0], path[i, 1], np.cos(headings[i]/10), np.sin(headings[i]/10),
                            head_width=0.5, head_length=1.5, linewidth=0, color='r')

    def show(self, mode=None, segment=None, which=None, layout=None, name=None, gui=False):
        plt.clf()
        if mode == 'range':
            plt.gca().set_aspect(1)
            plt.axis([self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]])
            for item in range(segment[0], segment[1]):
                if 'gridmap' in self.menu:
                    self.show_gridmap(item)
                if 'condition' in self.menu:
                    self.show_condition(item)
                if 'sproducts' in self.menu:
                    self.show_sproducts(item)
                if 'label' in self.menu:
                    self.show_label(item)
                plt.show()
                input("finish it {}?".format(item))
                plt.clf()
        if mode == 'list':
            plt.gca().set_aspect(1)
            plt.axis([self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]])
            for item in which:
                if 'gridmap' in self.menu:
                    self.show_gridmap(item)
                if 'condition' in self.menu:
                    self.show_condition(item)
                if 'sproducts' in self.menu:
                    self.show_sproducts(item)
                if 'label' in self.menu:
                    self.show_label(item)
                plt.show()
                input("finish it {}?".format(item))
                plt.clf()
        if mode == 'one':
            for i, item in enumerate(which):
                plt.subplot(layout + i)
                plt.xticks(fontsize=self.fontsize)
                plt.yticks(fontsize=self.fontsize)
                plt.gca().set_aspect(1)
                plt.axis([self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]])
                plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, right=0.9)
                if 'gridmap' in self.menu:
                    self.show_gridmap(item)
                if 'condition' in self.menu:
                    self.show_condition(item)
                if 'sproducts' in self.menu:
                    self.show_sproducts(item)
                if 'label' in self.menu:
                    self.show_label(item)
                if 'prediction' in self.menu:
                    self.show_prediction(item)
            plt.savefig(self.dir_parent + str(name) + '.svg')
            if gui:
                plt.show()
