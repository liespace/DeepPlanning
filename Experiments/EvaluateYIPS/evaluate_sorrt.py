#!/usr/bin/env python
import time
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import reeds_shepp
from matplotlib.patches import Polygon
from planner import RRTStar, BiRRTStar
from debugger import Debugger
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
    yips = [((yip[0], yip[1], yip[2]), ((0.131, 2.442), (0.071, 1.780 * 1.0), (-0.029, 0.507 * 1.0))) for yip in yips]
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
    elif 'checkpoint' in heuristic_name:
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


def main(seqs, dataset_folder, inputs_filename, heuristic_name, version,
         outputs_folder, outputs_tag, times, rounds, debug, optimize):
    outputs_folder = outputs_folder + os.sep + outputs_tag + os.sep + heuristic_name + os.sep + version
    os.makedirs(outputs_folder) if not os.path.isdir(outputs_folder) else None

    rrt_star = BiRRTStar().set_vehicle(contour(), 0.3, 0.2)
    for i, seq in enumerate(seqs):  # read_seqs(dataset_folder, inputs_filename)
        print('Processing Scene: {} ({} of {})'.format(seq, i+1, len(seqs)))
        heuristic = read_heuristic('./predictions'+os.sep+outputs_tag, seq, heuristic_name)
        source, target = read_task(dataset_folder+os.sep+'scenes', seq)
        start = center2rear(deepcopy(source)).gcs2lcs(source.state)
        goal = center2rear(deepcopy(target)).gcs2lcs(source.state)
        grid_ori = deepcopy(source).gcs2lcs(source.state)
        grid_map, grid_res = read_grid(dataset_folder+os.sep+'scenes', seq), 0.1
        rrt_star.debug = debug
        Debugger.plan_hist = []
        past = time.time()
        for r in range(rounds):
            rrt_star.preset(start, goal, grid_map, grid_res, grid_ori, 255, heuristic)
            rrt_star.planning(times, repeat=100, optimize=optimize, debug=debug)
            if rrt_star.x_best.fu < np.inf:
                path = rrt_star.path()
                np.savetxt('{}/{}_path.txt'.format(outputs_folder, seq), [p.state for p in path], delimiter=',')
        print ('    Runtime: {}'.format(time.time() - past))

        Debugger().save_hist(outputs_folder + os.sep + str(seq) + '_summary.txt')


def read_inference(filepath, seq, folder='vgg19_comp_free200_check300', discrimination=0.7):
    inference = np.loadtxt('{}/{}/{}_inference.txt'.format(filepath, folder, seq), delimiter=',')
    # inference = filter(lambda x: x[-1] > discrimination, inference)
    return inference


def read_label(dataset_folder, seq):
    label_filename = dataset_folder + os.sep + 'labels' + os.sep + '{}_path.txt'.format(seq)
    path = np.loadtxt(label_filename, delimiter=',')
    path = path[1:-1]
    label = np.zeros((5, 4))
    for r in range(path.shape[0]):
        label[r][:3], label[r][3] = path[r], 1
    return label


def build_path(start, sequence, goal, threshold):
    path = [list(start)]
    for sample in sequence:
        if sample[-1] > threshold:
            path.append(list(sample[:-1]))
    path.append(list(goal))
    return path


def calculate_path_length(path, rho):
    sequence = map(deepcopy, path)
    sequence = map(list, sequence)
    sequence = zip(sequence[:-1], sequence[1:])
    length = 0
    for x_from, x_to in sequence:
        length += reeds_shepp.path_length(x_from, x_to, rho)
    return length


def read_yips_planning_summary(filepath, seq, folder='vgg19_comp_free200_check300'):
    summary = np.loadtxt('{}/{}/{}_summary.txt'.format(filepath, folder, seq), delimiter=',')
    return summary


def new_figure(y_label='Precision[-]', x_label='Recall[-]', fontsize=55):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    return ax

def read_task2(filepath, seq=0):
    """
    read source(start) and target(goal), and transform to right-hand and local coordinate system centered in source
    LCS: local coordinate system, or said vehicle-frame.
    GCS: global coordinate system
    """
    # read task and transform coordinate system to right-hand
    task = np.loadtxt('{}/scenes/{}_task.txt'.format(filepath, seq), delimiter=',')
    org, aim = task[0], task[1]
    # coordinate of the center of mass on source(start) state, in GCS
    source = RRTStar.StateNode(state=(org[0], -org[1], -np.radians(org[3])))
    # coordinate of center of mass on target(goal) state, in GCS
    target = RRTStar.StateNode(state=(aim[0], -aim[1], -np.radians(aim[3])))
    start = center2rear(deepcopy(source)).gcs2lcs(source.state)
    goal = center2rear(deepcopy(target)).gcs2lcs(source.state)
    return start.state, goal.state


def calculate_performance(seqs, predictor, dataset_folder, inputs_filename, prediction_folder, planning_folder):
    threshold = 0.5
    rho = 5.0
    for i, seq in enumerate(seqs):  # enumerate(seqs)
        print('Evaluate Scene: {} ({} of {})'.format(seq, i + 1, len(seqs)))
        inference = read_inference(prediction_folder, seq, predictor)
        label = read_label(dataset_folder, seq)
        start, goal = read_task2(dataset_folder, seq)
        pred = build_path(start, inference, goal, threshold)
        true = build_path(start, label, goal, threshold)
        pred_length = calculate_path_length(pred, rho=rho)
        true_length = calculate_path_length(true, rho=rho)
        optimal_length = calculate_path_length([start, goal], rho=rho)

        fontsize = 40
        ax1 = new_figure(y_label='mLOP/LOOP', x_label='', fontsize=fontsize)
        # ax1.set_ylim([-0.05, 2.5])
        # ax1.set_xlim([0, 2750])
        # ax1.set_yticks(np.arange(0., 2.5, 0.5))
        # ax1.set_xticks([0, 1000, 2000])

        labels = ['SO-RRT', 'SO-RRT-OP', 'SO-RRT-GR']
        colors = ['r', 'b', 'g']
        for j, version in enumerate(['normal', 'optimal', 'greedy']):
            summary = read_yips_planning_summary(planning_folder, seq, predictor + os.sep + version)
            summaries = np.split(summary, 100)
            summary = np.mean(summaries, axis=0)

            lens = summary[:, 2]
            times = summary[:, 1]
            samples = summary[:, 0]
            normalized_lens = lens / optimal_length
            ax1.plot(times, normalized_lens, lw=6, c=colors[j], zorder=100 - j, label=labels[j])
            print true_length, optimal_length

        # ax1.hlines(true_length/optimal_length, 0, 500, lw=6, color='g', zorder=200, linestyles='dashed', label='Label')
        # ax1.hlines(optimal_length / optimal_length, 0, 500, lw=6, color='b', zorder=150, linestyles='dotted', label='Geodesic')
        ax1.legend(prop={'size': fontsize}, loc=4, frameon=True, ncol=2)
        plt.draw()
        plt.show()
        Debugger.breaker('')


if __name__ == '__main__':
    yips = 'rgous-vgg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr70_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-200'

    # seqs = [2384, 13269, 11036, 13590, 1095, 7412, 10930, 10955, 6045]
    sequences = [2384, 13269, 11036, 10955, 6045, 6025, 1000]
    # main(seqs=sequences,
    #      dataset_folder='../../DataMaker/dataset',  # ./Dataset
    #      inputs_filename='valid.csv',  # test.csv
    #      heuristic_name=yips,  # ose, none, yips
    #      outputs_folder='./sorrt_evaluation',
    #      outputs_tag='valid',
    #      version='greedy',
    #      times=500, rounds=100, debug=False, optimize=True)  # test

    calculate_performance(seqs=sequences,
                          predictor=yips, dataset_folder='../../DataMaker/dataset',
                          inputs_filename='valid.csv', prediction_folder='predictions/valid',
                          planning_folder='sorrt_evaluation/valid')

