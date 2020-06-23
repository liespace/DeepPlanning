#!/usr/bin/env python
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
import re
from matplotlib.patches import Polygon
from rrts.planner import RRTStar, BiRRTStar
from rrts.debugger import Debugger
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


def read_yips(filepath, seq, folder='vgg19_comp_free200_check300', discrimination=0.7):
    yips = np.loadtxt('{}/{}/{}_inference.txt'.format(filepath, folder, seq), delimiter=',')
    yips = filter(lambda x: x[-1] > discrimination, yips)
    yips = map(center2rear, yips)
    yips = [((yip[0], yip[1], yip[2]), ((0.621, 2.146), (0.015, 1.951 * 1.0), (0.005, 0.401 * 1.0))) for yip in yips]
    return yips


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


def plot_task(grid_map, grid_res, heuristic, start, goal):
    Debugger.plot_grid(grid_map, grid_res)
    Debugger.plot_heuristic(heuristic) if heuristic else None
    Debugger().plot_polygon(transform(contour(), start.state), color='y')
    Debugger().plot_polygon(transform(contour(), goal.state), color='y')
    plt.draw()


def transform(poly, pto):
    pts = poly.transpose()
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return (np.dot(rot, pts) + xyo).transpose()


def read_heuristic(heuristic_folder, seq, heuristic_name):
    if 'ose' in heuristic_name:
        return read_ose(heuristic_folder, seq, heuristic_name)  # read_ose
    elif 'yips' in heuristic_name:
        return read_yips(heuristic_folder, seq, heuristic_name)  # read_yips
    else:
        return None


def read_seqs(dataset_folder, inputs_filename):
    inputs_filepath = dataset_folder + os.sep + inputs_filename
    if 'csv' in inputs_filename:
        file_list = [f.rstrip().split(',')[0] for f in list(open(inputs_filepath))]
    else:
        file_list = os.listdir(inputs_filepath)
    return [re.sub('\\D', '', f.strip().split(',')[0]) for f in file_list]


def main(dataset_folder, heuristic_name, outputs_folder, times, rounds, debug, optimize):
    outputs_folder = dataset_folder + os.sep + outputs_folder + os.sep + heuristic_name
    rrt_star = BiRRTStar().set_vehicle(contour(), 0.3, 0.2)
    seqs = read_seqs(dataset_folder, 'inputs')
    for i, seq in enumerate([5668]):  # 5709 12736 5668, 10938 (need to change sigma of theta)
        print('Processing Scene: {}, {}/{}'.format(seq, i, len(seqs)))
        heuristic = read_heuristic('heuristics', seq, heuristic_name)
        source, target = read_task(dataset_folder+os.sep+'scenes', seq)
        start = center2rear(deepcopy(source)).gcs2lcs(source.state)
        goal = center2rear(deepcopy(target)).gcs2lcs(source.state)
        grid_ori = deepcopy(source).gcs2lcs(source.state)
        grid_map, grid_res = read_grid(dataset_folder+os.sep+'scenes', seq), 0.1
        Debugger.plan_hist, paths = [], []
        past = time.time()
        for r in range(rounds):
            rrt_star.preset(start, goal, grid_map, grid_res, grid_ori, 255, heuristic)
            rrt_star.planning(times, repeat=100, optimize=optimize, debug=debug)
            paths.append((rrt_star.x_best.fu, rrt_star.path()))
        runtime = time.time() - past
        fu, path = sorted(paths)[0]
        print('Runtime: {}, Result: {}'.format(runtime, fu < np.inf))
        os.makedirs(outputs_folder) if not os.path.isdir(outputs_folder) else None
        if fu < np.inf:
            np.savetxt('{}/{}_path.txt'.format(outputs_folder, seq), [p.state for p in path], delimiter=',')
            np.savetxt('{}/{}_length.txt'.format(outputs_folder, seq), [fu, rrt_star.start.fl], delimiter=',')


if __name__ == '__main__':
    main(dataset_folder='dataset',  # ./Dataset
         heuristic_name='ose',  # vgg19_comp_free200_check300, ose, none
         outputs_folder='plans',
         times=500, rounds=1, debug=False, optimize=False)  # test