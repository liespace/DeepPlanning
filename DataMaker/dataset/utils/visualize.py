#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import reeds_shepp


def center2rear(node, wheelbase=2.96):
    """calculate the coordinate of rear track center according to mass center"""
    theta, r = node[2] + np.pi, wheelbase / 2.
    node[0] += r * np.cos(theta)
    node[1] += r * np.sin(theta)
    return node


def contour(wheelbase=2.850, width=2.116 + 0.2, length=4.925 + 0.2):  # 2.96, 2.2, 5.0
    return np.array([
        [-(length/2. - wheelbase / 2.), width/2. - 1.0], [-(length/2. - wheelbase / 2. - 0.4), width/2.],
        [length/2. + wheelbase / 2. - 0.6, width/2.], [length/2. + wheelbase / 2., width/2. - 0.8],
        [length/2. + wheelbase / 2., -(width/2. - 0.8)], [length/2. + wheelbase / 2. - 0.6, -width/2.],
        [-(length/2. - wheelbase / 2. - 0.4), -width/2.], [-(length/2. - wheelbase / 2.), -(width/2. - 1.0)]])


def transform(poly, pto):
    pts = poly.transpose()
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return (np.dot(rot, pts) + xyo).transpose()


def plot_polygon(ploy, color='b', lw=2., fill=False):
    actor = plt.gca().add_patch(Polygon(ploy, True, color=color, fill=fill, linewidth=lw))
    plt.draw()
    return [actor]


def plot_polygons(x_from, x_to, rho, color='y'):
    states = reeds_shepp.path_sample(x_from, x_to, rho, 0.3)
    [plot_polygon(transform(contour(), s), color) for s in states]


def plot_curve(x_from, x_to, rho, color='g'):
    states = reeds_shepp.path_sample(x_from, x_to, rho, 0.3)
    x, y = [state[0] for state in states], [state[1] for state in states]
    actor = plt.plot(x, y, c=color)
    plt.draw()
    return actor


def plot_state(state, color=(0.5, 0.8, 0.5)):
    cir = plt.Circle(xy=(state[0], state[1]), radius=0.4, color=color, alpha=0.6)
    arr = plt.arrow(x=state[0], y=state[1], dx=1.5 * np.cos(state[2]), dy=1.5 * np.sin(state[2]), width=0.2)
    actors = [plt.gca().add_patch(cir), plt.gca().add_patch(arr)]
    plt.draw()
    return actors


def plot_grid(grid_map, grid_res):
    # type: (np.ndarray, float) -> None
    """plot grid map"""
    row, col = grid_map.shape[0], grid_map.shape[1]
    indexes = np.argwhere(grid_map == 255)
    xy2uv = np.array([[0., 1. / grid_res, row / 2.], [1. / grid_res, 0., col / 2.], [0., 0., 1.]])
    for index in indexes:
        uv = np.array([index[0], index[1], 1])
        xy = np.dot(np.linalg.inv(xy2uv), uv)
        rect = plt.Rectangle((xy[0] - grid_res, xy[1] - grid_res), grid_res, grid_res, color=(1.0, 0.1, 0.1))
        plt.gca().add_patch(rect)
    plt.draw()


def plot_path(path, rho=5.):
    path = map(list, path)
    print(path)
    # path = [center2rear(p) for p in path]
    print(path)
    start, goal = path[0], path[-1]
    [plot_state(p) for p in path]
    path = zip(path[:-1], path[1:])
    [plot_curve(x_from, x_to, rho, 'r') for x_from, x_to in path]
    [plot_polygons(x_from, x_to, rho) for x_from, x_to in path]
    plot_polygon(transform(contour(), start))
    plot_polygon(transform(contour(), goal))


def set_plot(switch=True):
    if switch:
        plt.ion()
        plt.figure()
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().set_aspect('equal')
        plt.gca().set_facecolor((0.2, 0.2, 0.2))
        plt.gca().set_xlim((-30, 30))
        plt.gca().set_ylim((-30, 30))
        plt.draw()


def main(no=0, labels_folder='labels', scenes_folder='scenes'):
    set_plot()
    path_filename = labels_folder + os.sep + str(no) + '_path.txt'
    grid_filename = scenes_folder + os.sep + str(no) + '_gridmap.png'
    path = np.loadtxt(path_filename, delimiter=',')
    grid = cv2.imread(grid_filename, -1)
    plot_path(path)
    plot_grid(grid, 0.1)


if __name__ == '__main__':
    main(no=5668)
    raw_input('Plotting')
