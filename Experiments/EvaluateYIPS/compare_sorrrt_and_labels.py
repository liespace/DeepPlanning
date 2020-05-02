#!/usr/bin/env python
import time
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
from matplotlib.patches import Polygon
from rrts.planner import RRTStar, BiRRTStar
from rrts.debugger import Debugger
import reeds_shepp
import matplotlib.lines as mlines
import logging


vehicle_wheelbase = 2.850
vehicle_length = 4.925 + 0.2
vehicle_width = 2.116 + 0.2


def center2rear(node, wheelbase=2.850):
    """calculate the coordinate of rear track center according to mass center"""
    if not isinstance(node, RRTStar.StateNode):
        theta, r = node[2] + np.pi, wheelbase / 2.
        node[0] += r * np.cos(theta)
        node[1] += r * np.sin(theta)
        return node
    theta, r = node.state[2] + np.pi, wheelbase / 2.
    node.state[0] += r * np.cos(theta)
    node.state[1] += r * np.sin(theta)
    return node


def contour(wheelbase=2.850, width=2.116 + 0.2, length=4.925 + 0.2):  # 2.96, 2.2, 5.0
    return np.array([
        [-(length/2. - wheelbase / 2.), width/2. - 1.0], [-(length/2. - wheelbase / 2. - 0.4), width/2.],
        [length/2. + wheelbase / 2. - 0.6, width/2.], [length/2. + wheelbase / 2., width/2. - 0.8],
        [length/2. + wheelbase / 2., -(width/2. - 0.8)], [length/2. + wheelbase / 2. - 0.6, -width/2.],
        [-(length/2. - wheelbase / 2. - 0.4), -width/2.], [-(length/2. - wheelbase / 2.), -(width/2. - 1.0)]])


def read_task(filepath, seq=0):
    """
    read source(start) and target(goal), and transform to right-hand and local coordinate system centered in source
    LCS: local coordinate system, or said vehicle-frame.
    GCS: global coordinate system
    """
    # read task and transform coordinate system to right-hand
    task = np.loadtxt('{}/{}_task.txt'.format(filepath, seq), delimiter=',')
    org, aim = task[0], task[1]
    # coordinate of the center of mass on source(start) state, in GCS
    source = RRTStar.StateNode(state=(org[0], -org[1], -np.radians(org[3])))
    # coordinate of center of mass on target(goal) state, in GCS
    target = RRTStar.StateNode(state=(aim[0], -aim[1], -np.radians(aim[3])))
    return source, target


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return cv2.imread(filename='{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


def read_ose(filepath, seq, folder='ose'):
    """read heuristic ose"""
    oseh = np.loadtxt('{}/{}/{}_corridor.txt'.format(filepath, folder, seq), delimiter=',')
    oseh = [((x[0], x[1], x[2]), ((0., x[3]/3.), (0., x[3]/3.), (0., np.pi/4./1.))) for x in oseh]
    return oseh


def read_yips(filepath, seq, folder='vgg19_comp_free200_check300', discrimination=0.5):
    yips = np.loadtxt('{}/{}/{}_inference.txt'.format(filepath, folder, seq), delimiter=',')
    yips = filter(lambda x: x[-1] > discrimination, yips)
    # yips = map(center2rear, yips)
    yips = [((yip[0], yip[1], yip[2]), ((0.621, 2.146), (0.015, 1.951 * 1.0), (0.005, 0.401 * 1.0))) for yip in yips]
    return yips


def plot_grid(grid_map, grid_res, color='k'):
    """plot grid map"""
    row, col = grid_map.shape[0], grid_map.shape[1]
    indexes = np.argwhere(grid_map == 255)
    xy2uv = np.array([[0., 1. / grid_res, row / 2.], [1. / grid_res, 0., col / 2.], [0., 0., 1.]])
    for index in indexes:
        uv = np.array([index[0], index[1], 1])
        xy = np.dot(np.linalg.inv(xy2uv), uv)
        rect = plt.Rectangle((xy[0] - grid_res/2., xy[1] - grid_res/2.), grid_res, grid_res, color=color)
        plt.gca().add_patch(rect)
    plt.draw()


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


def plot_config(state, color='y', lw=3., zorder=10):
    plot_polygon(transform(contour(), state), color=color, lw=lw, zorder=zorder)
    pt0, pt1, pt2, pt3 = deepcopy(state), deepcopy(state), deepcopy(state), deepcopy(state)

    def trans_pt(pt, delta, r):
        theta = pt[2] + delta
        pt[0] += r * np.cos(theta)
        pt[1] += r * np.sin(theta)
        return pt
    pt0 = trans_pt(pt0, 0, 3.)
    pt1 = trans_pt(pt1, np.pi, 0.8)
    pt2 = trans_pt(pt2, np.pi/2, 0.8)
    pt3 = trans_pt(pt3, -np.pi/2, 0.8)
    plt.gca().add_line(mlines.Line2D([pt0[0], pt1[0]], [pt0[1], pt1[1]], color=color, linewidth=lw, zorder=zorder, markeredgecolor='k'))
    plt.gca().add_line(mlines.Line2D([pt2[0], pt3[0]], [pt2[1], pt3[1]], color=color, linewidth=lw, zorder=zorder, markeredgecolor='k'))
    plt.draw()


def plot_task(grid_map, grid_res, heuristic, start, goal):
    plot_grid(grid_map, grid_res, color='C1')
    # Debugger.plot_heuristic(heuristic) if heuristic else None
    plt.draw()


def transform(poly, pto):
    pts = poly.transpose()
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return (np.dot(rot, pts) + xyo).transpose()


def read_heuristic(heuristic_folder, seq, heuristic_name):
    if 'ose' in heuristic_name:
        return read_ose(heuristic_folder, seq, heuristic_name)  # read_ose
    else:
        return read_yips(heuristic_folder, seq, heuristic_name)  # read_yips


def read_seqs(dataset_folder, inputs_filename):
    inputs_filepath = dataset_folder + os.sep + inputs_filename
    if 'csv' in inputs_filename:
        file_list = [f.rstrip().split(',')[0] for f in list(open(inputs_filepath))]
    else:
        file_list = os.listdir(inputs_filepath)
    return [re.sub('\\D', '', f.strip().split(',')[0]) for f in file_list]


def plot_state(state, color='C0'):
    ratio = 1.5
    arr = plt.arrow(x=state[0], y=state[1], dx=1.5 * np.cos(state[2])*ratio, dy=1.5 * np.sin(state[2])*ratio, width=0.3*ratio, zorder=100)
    cir = plt.Circle(xy=(state[0], state[1]), radius=0.5*ratio, ec='k', zorder=1000)
    actors = [plt.gca().add_patch(arr), plt.gca().add_patch(cir)]
    plt.draw()
    return actors


def plot_polygon(ploy, color='b', lw=2., fill=False, zorder=10):
    actor = plt.gca().add_patch(Polygon(ploy, True, color=color, fill=fill, linewidth=lw, zorder=zorder))
    plt.draw()
    return [actor]


def plot_polygons(x_from, x_to, rho, color='y', lw=3., zorder=10):
    states = reeds_shepp.path_sample(x_from, x_to, rho, 0.2)
    [plot_polygon(transform(contour(), s), color, lw=lw, zorder=zorder) for s in states]


def plot_curve(x_from, x_to, rho, color='g', lw=3., zorder=10):
    states = reeds_shepp.path_sample(x_from, x_to, rho, 0.3)
    x, y = [state[0] for state in states], [state[1] for state in states]
    actor = plt.plot(x, y, c=color, lw=lw, zorder=zorder)
    plt.draw()
    return actor


def plot_path(path, rho=5., real=False, lw=3., curve_color='C2', contour_color='C0', zorder=10):
    sequence = map(list, path)
    segments = zip(sequence[:-1], sequence[1:])
    [plot_curve(x_from, x_to, rho, lw=lw, color=curve_color, zorder=zorder+10) for x_from, x_to in segments]
    if real:
        [plot_polygons(x_from, x_to, rho, color=contour_color, lw=lw*0.5, zorder=zorder) for x_from, x_to in segments]


def main(dataset_folder, inputs_filename, heuristic_name, folder):
    seqs = read_seqs(dataset_folder, inputs_filename)
    seqs.sort()
    # iov == 0: [6823, 2384, 10902, 11210, 13500, 164]
    # iov <= 0.125: [13269, 1120, 2724, 9485]
    # iov > 0.125: [1000, 8157, 6025]
    for i, seq in enumerate([6045]):  # enumerate(seqs)
        print('Processing Scene: {} ({} of {})'.format(seq, i+1, len(seqs)))
        heuristic = read_heuristic(folder, seq, heuristic_name)
        source, target = read_task(dataset_folder+os.sep+'scenes', seq)
        start = center2rear(deepcopy(source)).gcs2lcs(source.state)
        goal = center2rear(deepcopy(target)).gcs2lcs(source.state)
        grid_map, grid_res = read_grid(dataset_folder+os.sep+'scenes', seq), 0.1
        path_filename = dataset_folder + os.sep + 'labels' + os.sep + '{}_path.txt'.format(seq)
        path = np.loadtxt(path_filename, delimiter=',')
        predicted_path = []
        for h in heuristic:
            predicted_path.append(list(h[0]))
        print('Labeled Samples: {}'.format([list(p) for p in path[1:-1]]))
        predicted_path.append(list(path[-1]))
        predicted_path.insert(0, list(path[0]))
        print('Predicted Samples: {}'.format([list(p) for p in predicted_path[1:-1]]))
        planned_path = np.loadtxt('./sorrt_evaluation/valid/{}/optimal/{}_path.txt'.format(heuristic_name, seq), delimiter=',')
        print('Planned Samples: {}'.format([list(p) for p in path[1:-1]]))

        set_plot()
        plot_task(grid_map, grid_res, heuristic, start, goal)
        plot_path(path, rho=5., real=False, lw=15., curve_color='g', contour_color='C0', zorder=50)
        plot_path(planned_path, rho=5., real=False, lw=15., curve_color='r', contour_color='C0', zorder=200)
        plot_path(predicted_path, rho=5., real=False, lw=15, curve_color='b', contour_color='g', zorder=20)
        # [plot_state(p) for p in predicted_path[1:-1]]
        # [plot_config(p, color='r', lw=5.*1.5, zorder=200) for p in planned_path[1:-1]]
        # [plot_config(p, color='g', lw=5.*1.5, zorder=20) for p in predicted_path[1:-1]]
        plot_config(start.state, color='y', lw=5.*1.5, zorder=10)
        plot_config(goal.state, color='y', lw=5.*1.5, zorder=10)
        Debugger.breaker('Plotting')


if __name__ == '__main__':
    predictors = [
        'rgous-vgg19v1C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine150[]_wp0o0e+00)-checkpoint-150',
        'rgous-vgg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr70_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-200',
        'rgous-vgg19C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine200[])-checkpoint-200',
        'rgous-res50PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr30_steps10[30, 140, 170]_wp0o0e+00)-checkpoint-200',
        'rgous-svg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-150',
        'rgous-svg16v1PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[75, 95, 135]_wp0o0e+00)-checkpoint-200',
        'rgous-vgg19v2C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_steps10[75, 105, 135]_wp0o0e+00)-checkpoint-200']

    main(dataset_folder='../../DataMaker/dataset',  # ./Dataset
         inputs_filename='valid.csv',  # test.csv
         heuristic_name=predictors[1],  # vgg19_comp_free200_check300, ose, none
         folder='predictions/valid')  # test

    # 'rgous-vgg19v1C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine150[]_wp0o0e+00)',
    # 'rgous-vgg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr70_steps10[70, 95, 110]_wp0o0e+00)',
    # 'rgous-vgg19C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine200[])',
    # 'rgous-res50PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr30_steps10[30, 140, 170]_wp0o0e+00)',
    # 'rgous-svg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[70, 95, 110]_wp0o0e+00)',
    # 'rgous-vgg19v2C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_steps10[75, 105, 135]_wp0o0e+00)-checkpoint-200'
