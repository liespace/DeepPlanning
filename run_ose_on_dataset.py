#!/usr/bin/env python
from pySEA.explorers import OrientationSpaceExplorer as OSExplorer
from copy import deepcopy
import time
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt


def center2rear(circle, wheelbase=2.850):  # type: (OSExplorer.CircleNode, float) -> OSExplorer.CircleNode
    """calculate the coordinate of rear track center according to mass center"""
    theta, r = circle.a + np.pi, wheelbase/2.
    circle.x += r * np.cos(theta)
    circle.y += r * np.sin(theta)
    return circle


def read_inputs(filepath, size=(480, 480), dataset_folder='./Dataset'):
    x = [cv2.imread(dataset_folder + x_path) for x_path in filepath]
    x = [cv2.resize(i, size) for i in x]
    return np.array(x)


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
    source = OSExplorer.CircleNode(x=org[0], y=-org[1], a=-np.radians(org[3]))
    # coordinate of center of mass on target(goal) state, in GCS
    target = OSExplorer.CircleNode(x=aim[0], y=-aim[1], a=-np.radians(aim[3]))
    return source, target


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return cv2.imread(filename='{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


def set_plot(explorer):
    # type: (OSExplorer) -> None
    plt.ion()
    plt.figure()
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_aspect('equal')
    plt.gca().set_facecolor((0.2, 0.2, 0.2))
    plt.gca().set_xlim((-30, 30))
    plt.gca().set_ylim((-30, 30))
    explorer.plot_grid(explorer.grid_map, explorer.grid_res)
    explorer.plot_circles([explorer.start, explorer.goal])
    plt.draw()


def plotting(explorer, debug=False):
    if debug:
        set_plot(explorer)
        explorer.plot_circles(explorer.circle_path)
        plt.draw()
        raw_input('Plotting')


def read_seqs(dataset_folder, inputs_filename):
    inputs_filepath = dataset_folder + os.sep + inputs_filename
    if 'csv' in inputs_filename:
        file_list = [f.rstrip().split(',')[0] for f in list(open(inputs_filepath))]
    else:
        file_list = os.listdir(inputs_filepath)
    return [re.sub('\\D', '', f.strip().split(',')[0]) for f in file_list]


def main(dataset_folder, inputs_filename, outputs_tag, rounds):
    outputs_folder = './predictions' + os.sep + outputs_tag + os.sep + 'ose'
    explorer = OSExplorer(
        minimum_radius=0.200,
        maximum_radius=3.029,  # 2.5
        minimum_clearance=1.058,
        neighbors=32,
        maximum_curvature=0.200)
    seqs = read_seqs(dataset_folder, inputs_filename)
    for i, seq in enumerate(seqs):
        print('Processing Scene: {} ({} of {})'.format(seq, i+1, len(seqs)))
        source, target = read_task(dataset_folder + '/scenes', seq)
        start = center2rear(deepcopy(source)).gcs2lcs(source)  # coordinate of rear track center on start state in LCS
        goal = center2rear(deepcopy(target)).gcs2lcs(source)  # coordinate of rear track center on goal state in LCS
        grid_ori = deepcopy(source).gcs2lcs(source)  # coordinate of grid map center in LCS
        grid_map, grid_res = read_grid(dataset_folder + '/scenes', seq), 0.1
        explorer.initialize(start, goal, grid_map, grid_res, grid_ori, obstacle=255)
        map(explorer.exploring, [None])  # compile jit

        def predicting(x):
            past = time.time()
            return explorer.exploring(), time.time() - past
        result, runtime = zip(*map(predicting, range(rounds)))
        print('    Runtime: {}s, SR: {}'.format(np.sum(runtime), result[-1]))

        os.makedirs(outputs_folder) if not os.path.isdir(outputs_folder) else None
        np.savetxt('{}/{}_corridor.txt'.format(outputs_folder, seq), explorer.path(), delimiter=',')
        np.savetxt('{}/{}_summary.txt'.format(outputs_folder, seq), [result, runtime], delimiter=',')


if __name__ == '__main__':
    main(dataset_folder='Dataset', inputs_filename='test.csv', outputs_tag='test', rounds=10)
