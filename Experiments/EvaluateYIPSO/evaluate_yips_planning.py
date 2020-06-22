#!/usr/bin/env python
from copy import deepcopy
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
from rrts.planner import RRTStar
import reeds_shepp


def transform(poly, pto):
    pts = poly.transpose()
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return (np.dot(rot, pts) + xyo).transpose()


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


def contour(wheelbase=2.850, width=2.116 + 0.2, length=4.925 + 0.2):  # 2.96, 2.2, 4.925
    return np.array([
        [-(length / 2. - wheelbase / 2.), width / 2. - 1.0], [-(length / 2. - wheelbase / 2. - 0.4), width / 2.],
        [length / 2. + wheelbase / 2. - 0.6, width / 2.], [length / 2. + wheelbase / 2., width / 2. - 0.8],
        [length / 2. + wheelbase / 2., -(width / 2. - 0.8)], [length / 2. + wheelbase / 2. - 0.6, -width / 2.],
        [-(length / 2. - wheelbase / 2. - 0.4), -width / 2.], [-(length / 2. - wheelbase / 2.), -(width / 2. - 1.0)]])


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return cv2.imread('{}/scenes/{}_gridmap.png'.format(filepath, seq), flags=-1)


def read_inference(filepath, seq, folder='vgg19_comp_free200_check300', discrimination=0.7):
    inference = np.loadtxt('{}/{}/{}_inference.txt'.format(filepath, folder, seq), delimiter=',')
    # inference = filter(lambda x: x[-1] > discrimination, inference)
    return inference


def read_summary(filepath, folder='vgg19_comp_free200_check300'):
    summary = []
    with open('{}/{}/0summary.csv'.format(filepath, folder)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            summary.append(row)
    return summary


def read_seqs(dataset_folder, inputs_filename):
    inputs_filepath = dataset_folder + os.sep + inputs_filename
    if 'csv' in inputs_filename:
        file_list = [f.rstrip().split(',')[0] for f in list(open(inputs_filepath))]
    else:
        file_list = os.listdir(inputs_filepath)
    return [re.sub('\\D', '', f.strip().split(',')[0]) for f in file_list]


def read_task(filepath, seq=0):
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


def read_label(dataset_folder, seq):
    label_filename = dataset_folder + os.sep + 'labels' + os.sep + '{}_path.txt'.format(seq)
    path = np.loadtxt(label_filename, delimiter=',')
    path = path[1:-1]
    label = np.zeros((5, 4))
    for r in range(path.shape[0]):
        label[r][:3], label[r][3] = path[r], 1
    return label


def trimmed_prediction_and_ground_truth(ground_truth, prediction):
    trimmed_prediction, trimmed_ground_truth = [], []
    for i in range(len(ground_truth)):
        if i % 5 == 0:
            continue
        trimmed_ground_truth.append(ground_truth[i])
        trimmed_prediction.append(prediction[i])
    return trimmed_ground_truth, trimmed_prediction


def trimmed_ious(ious):
    tri_ious = []
    for i in range(len(ious)):
        if i % 5 == 0:
            continue
        tri_ious.append(ious[i])
    return tri_ious


def new_figure(y_label='Precision[-]', x_label='Recall[-]', fontsize=55):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    return ax


def calculate_inference_time(predictor, prediction_folder):
    summary = read_summary(prediction_folder, predictor)
    infer_times = [float(row[1]) for row in summary]
    return np.mean(infer_times)*1000


def build_path(start, sequence, goal, threshold):
    path = [list(start)]
    for sample in sequence:
        if sample[-1] > threshold:
            path.append(list(sample[:-1]))
    path.append(list(goal))
    return path


def calculate_vehicle_area():
    veh_con = transform(contour(), [0, 0, 0])
    veh_con = np.floor(veh_con / 0.1 + 600 / 2.).astype(int)
    veh_mask = np.zeros((600, 600), dtype=np.uint8)
    cv2.fillPoly(veh_mask, [veh_con], 255)
    vehicle_area = np.sum(veh_mask / 255)
    return vehicle_area


def collision_check(path, rho, grid_map, iov_threshold=0.125):
    sequence = map(deepcopy, path)
    sequence = map(list, sequence)
    sequence = zip(sequence[:-1], sequence[1:])
    free, intersection_area = True, 0
    for x_from, x_to in sequence:
        states = reeds_shepp.path_sample(x_from, x_to, rho, 0.1)
        cons = [transform(contour(), state) for state in states]
        cons = [np.floor(con / 0.1 + 600 / 2.).astype(int) for con in cons]
        mask = np.zeros_like(grid_map, dtype=np.uint8)
        [cv2.fillPoly(mask, [con], 255) for con in cons]
        intersection = np.bitwise_and(mask, grid_map)
        free *= np.all(intersection < 255)
        intersection_area += np.sum(intersection / 255)
    iov = 1. * intersection_area / calculate_vehicle_area()
    return (False, iov) if not free and iov > iov_threshold else (True, iov)


def calculate_path_length(path, rho):
    sequence = map(deepcopy, path)
    sequence = map(list, sequence)
    sequence = zip(sequence[:-1], sequence[1:])
    length = 0
    for x_from, x_to in sequence:
        length += reeds_shepp.path_length(x_from, x_to, rho)
    return length


def calculate_performance(predictor, dataset_folder, inputs_filename, prediction_folder):
    inference_time = calculate_inference_time(predictor, prediction_folder)
    seqs = read_seqs(dataset_folder, inputs_filename)
    seqs.sort()
    threshold = 0.5
    iov_threshold = 0.125*0
    rho = 5.0
    pred_path_lens, true_path_lens, optimal_path_lens, collision_check_results, iovs = [], [], [], [], []
    for i, seq in enumerate(seqs):
        # print('Evaluate Scene: {} ({} of {})'.format(seq, i + 1, len(seqs)))
        inference = read_inference(prediction_folder, seq, predictor)
        label = read_label(dataset_folder, seq)
        start, goal = read_task(dataset_folder, seq)
        grid_map, grid_res = read_grid(dataset_folder, seq), 0.1
        pred = build_path(start, inference, goal, threshold)
        true = build_path(start, label, goal, threshold)

        pred_length = calculate_path_length(pred, rho=rho)
        true_length = calculate_path_length(true, rho=rho)
        optimal_length = calculate_path_length([start, goal], rho=rho)
        result, iov = collision_check(pred, rho, grid_map, iov_threshold=iov_threshold)
        # print('IOV={}'.format(iov))
        pred_path_lens.append(pred_length)
        true_path_lens.append(true_length)
        optimal_path_lens.append(optimal_length)
        collision_check_results.append(result)
        iovs.append(iov)
    print ("Evaluate {}".format(predictor))
    print ('mIT: {:.2f}'.format(inference_time))
    print ('mLOP: {}/{}'.format(np.mean(pred_path_lens), np.mean(pred_path_lens)-np.mean(true_path_lens)))
    print ('SR: {} (IOV={})'.format(1.*np.sum(collision_check_results)/len(collision_check_results), iov_threshold))
    print ('mIOV/IOV_max: {}/{}'.format(np.array(iovs).mean(), max(iovs)))


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    yips = [
        'rgous-vgg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr70_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-200',
        'rgous-res50PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr30_steps10[30, 140, 170]_wp0o0e+00)-checkpoint-200',
        'rgous-svg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-150',
        'rgous-vgg19v2C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_steps10[75, 105, 135]_wp0o0e+00)-checkpoint-200']
    calculate_performance(
        predictor=yips[-1], dataset_folder='../../DataMaker/dataset',
        inputs_filename='valid.csv', prediction_folder='predictions/valid')
