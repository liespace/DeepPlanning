#!/usr/bin/env python
import time
from copy import deepcopy
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
from matplotlib.patches import Polygon
import matplotlib.lines as mlines
from planner import RRTStar, BiRRTStar
from debugger import Debugger
import reeds_shepp
import logging
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from matplotlib.ticker import FormatStrFormatter


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


def plot_polygon(ploy, color='b', lw=2., fill=False):
    actor = plt.gca().add_patch(Polygon(ploy, True, color=color, fill=fill, linewidth=lw))
    plt.draw()
    return [actor]


def plot_state2(state, color='y'):
    plot_polygon(transform(contour(), state), color=color)
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
    plt.gca().add_line(mlines.Line2D([pt0[0], pt1[0]], [pt0[1], pt1[1]], color=color, linewidth=2))
    plt.gca().add_line(mlines.Line2D([pt2[0], pt3[0]], [pt2[1], pt3[1]], color=color, linewidth=2))
    plt.draw()


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


def plot_path(path, rho=5., real=False, color='r'):
    path = map(list, path)
    [plot_state2(p, color) for p in path[1:-1]]
    path = zip(path[:-1], path[1:])
    [plot_curve(x_from, x_to, rho, color) for x_from, x_to in path]
    if real:
        [plot_polygons(x_from, x_to, rho, color) for x_from, x_to in path]


def plot_task(grid_map, grid_res, start, goal):
    Debugger.plot_grid(grid_map, grid_res)
    plot_state2(start)
    plot_state2(goal)
    plt.draw()


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


def rectangle(wheelbase=2.850, width=2.116 + 0.2, length=4.925 + 0.2):  # 2.96, 2.2, 5.0
    return np.array([
        [-(length / 2. - wheelbase / 2.), width / 2.], [length / 2. + wheelbase / 2., width / 2.],
        [length / 2. + wheelbase / 2., -width / 2.], [-(length / 2. - wheelbase / 2.), -width / 2.]])


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return cv2.imread('{}/scenes/{}_gridmap.png'.format(filepath, seq), flags=-1)


def calculate_iou(inferences, labels):
    ious = []
    for i in range(inferences.shape[0]):
        pred = inferences[i][:3]
        true = labels[i][:3]
        pred_con = transform(rectangle(), pred)
        true_con = transform(rectangle(), true)
        pred_con = np.floor(pred_con / 0.1 + 600 / 2.).astype(int)
        true_con = np.floor(true_con / 0.1 + 600 / 2.).astype(int)
        pred_mask = np.zeros((600, 600), dtype=np.uint8)
        true_mask = np.zeros((600, 600), dtype=np.uint8)
        cv2.fillPoly(pred_mask, [pred_con], 255)
        cv2.fillPoly(true_mask, [true_con], 255)
        true_and_pred = np.bitwise_and(true_mask, pred_mask)
        true_or_pred = np.bitwise_or(true_mask, pred_mask)
        ious.append(np.sum(true_and_pred / 255.) / np.sum(true_or_pred / 255.))
    return ious


def calculate_iou_with_obstacle(inferences, dataset_folder, inputs_filename):
    seqs = read_seqs(dataset_folder, inputs_filename)
    seqs.sort()
    ious = []
    for i in range(inferences.shape[0]):
        # print 'Scene: {} @ {}/{}'.format(seqs[i/5], i/5, len(seqs))
        pred = inferences[i][:3]
        pred_con = transform(contour(), pred)
        pred_con = np.floor(pred_con / 0.1 + 600 / 2.).astype(int)
        pred_mask = np.zeros((600, 600), dtype=np.uint8)
        cv2.fillPoly(pred_mask, [pred_con], 255)
        grid_map = read_grid(dataset_folder + os.sep + 'scenes', seqs[i / 5])
        grid_map /= 255
        grid_map *= 255
        pred_and_obstacle = np.bitwise_and(grid_map, pred_mask)
        ious.append(np.sum(pred_and_obstacle / 255.) / np.sum(pred_mask / 255.))
    return ious


def calculate_iou_pr_and_ap():
    inferences = np.loadtxt('inferences.txt', delimiter=',')
    labels = np.loadtxt('labels.txt', delimiter=',')
    ground_truth = list(labels[:, -1])
    prediction = inferences[:, -1]
    threshold = np.unique(prediction.round(8))
    threshold = filter(lambda x: x > 1e-3, threshold)
    print(np.array(threshold))
    print(len(threshold))
    ious = calculate_iou(inferences, labels)


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

def read_ose_summary(filepath):
    summary = np.loadtxt('{}/{}/summary.txt'.format(filepath, 'ose'), delimiter=',')
    return summary


def read_yips_planning_summary(filepath, seq, folder='vgg19_comp_free200_check300'):
    summary = np.loadtxt('{}/{}/{}_summary.txt'.format(filepath, folder, seq), delimiter=',')
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


def extract_prediction_and_ground_truth(dataset_folder, inputs_filename, predictor, folder):
    seqs = read_seqs(dataset_folder, inputs_filename)
    seqs.sort()
    prediction, ground_truth = [], []
    inferences, labels = [], []
    for i, seq in enumerate(seqs):  # enumerate(seqs)
        print('Evaluate Scene: {} ({} of {})'.format(seq, i + 1, len(seqs)))
        inference = read_inference(folder, seq, predictor)
        label = read_label(dataset_folder, seq)
        prediction.extend(list(inference[:, -1]))
        ground_truth.extend(list(label[:, -1]))
        inferences.extend([list(p) for p in inference])
        labels.extend([list(p) for p in label])
        # print('Inference:\n {}'.format(inference))
        # print('Label:\n {}'.format(label))
        # break
    # print('Prediction:\n {}'.format(prediction))
    # print('GroundTruth:\n {}'.format(ground_truth))
    print('NumberOfSamples: {}/{}'.format(sum(ground_truth), len(ground_truth)))
    # np.savetxt('necessity_prediction.txt', prediction, delimiter=',')
    # np.savetxt('necessity_ground_truth.txt', ground_truth, delimiter=',')
    np.savetxt('inferences-{}.txt'.format(predictor), inferences, delimiter=',')
    np.savetxt('labels.txt', labels, delimiter=',')


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
    return np.mean(infer_times)


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
    # cv2.imshow('vehicle', veh_mask)
    # cv2.waitKey(0)
    return vehicle_area


def collision_check(path, rho, grid_map, iov_threshold=0.125):
    sequence = map(deepcopy, path)
    sequence = map(list, sequence)
    sequence = zip(sequence[:-1], sequence[1:])
    free, intersection_area = True, 0
    for x_from, x_to in sequence:
        states = reeds_shepp.path_sample(x_from, x_to, rho, 0.3)
        cons = [transform(contour(), state) for state in states]
        cons = [np.floor(con / 0.1 + 600 / 2.).astype(int) for con in cons]
        mask = np.zeros_like(grid_map, dtype=np.uint8)
        [cv2.fillPoly(mask, [con], 255) for con in cons]
        intersection = np.bitwise_and(mask, grid_map)
        free *= np.all(intersection < 255)
        intersection_area += np.sum(intersection / 255)

        # cv2.imshow('intersection', intersection)
        # cv2.imshow('Union', np.bitwise_or(mask, grid_map))
        # cv2.imshow('vehicle', veh_mask)
        # cv2.waitKey(0)
        # print iov, intersection_area, vehicle_area
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


def calculate_ose_inference_time(prediction_folder):
    summary = read_ose_summary(prediction_folder)
    inference_times = summary[:, 0].mean()
    return inference_times


def compare_inference_time(predictor, prediction_folder):
    ose_summary = read_ose_summary(prediction_folder)
    ose_inference_times = ose_summary[:, 0]
    yips_summary = read_summary(prediction_folder, predictor)
    yips_infer_times = [float(row[1]) for row in yips_summary]

    print min(ose_inference_times), min(yips_infer_times)

    data = zip(ose_inference_times, yips_infer_times)
    # data.sort()
    data = np.array(data)
    fontsize = 40
    ax1 = new_figure(y_label='Inference Time [ms]', x_label='Scene Number', fontsize=fontsize)
    ax1.set_yscale('log')
    # ax1.set_ylim([-0.05, 2.5])
    # ax1.set_xlim([0, 2750])
    # ax1.set_yticks(np.arange(0., 2.5, 0.5))
    # ax1.set_xticks([0, 1000, 2000])
    ax1.plot(data[:, 0], lw=4, c='C2', zorder=50, label='OSE')
    ax1.plot(data[:, 1], lw=4, c='r', zorder=100, label='YIPS')
    ax1.legend(prop={'size': fontsize}, loc=0, frameon=True, ncol=2, framealpha=1).set_zorder(102)
    plt.show()


def extract_yips_summary(predictor, dataset_folder, inputs_filename, prediction_folder, planning_folder, max_samples=500):
    summary = read_summary(prediction_folder, predictor)
    infer_times = [(row[0].split('_')[0], float(row[1])) for row in summary]
    infer_times.sort()
    seqs = read_seqs(dataset_folder, inputs_filename)
    seqs.sort()
    threshold, rho = 0.5, 5.0
    SPIT = 0.
    STFPs, LOFPs, LOLPs, TTFP, Fails, LOnPs, LOtPs = [], [], [], [], [], [], []
    wSTFPs, wLOFPs, wLOLPs, wTTFPs, wLOnPs, wLOtPs = [], [], [], [], [], []
    pred_path_lens, true_path_lens, optimal_path_lens = [], [], []
    all_pred_path_lens, all_true_path_lens, all_optimal_path_lens = [], [], []
    for i, seq in enumerate(seqs):  # enumerate(seqs)
        # print('Evaluate Scene: {} ({} of {})'.format(seq, i + 1, len(seqs)))
        inference = read_inference(prediction_folder, seq, predictor)
        label = read_label(dataset_folder, seq)
        start, goal = read_task(dataset_folder, seq)
        pred = build_path(start, inference, goal, threshold)
        true = build_path(start, label, goal, threshold)
        pred_length = calculate_path_length(pred, rho=rho)
        true_length = calculate_path_length(true, rho=rho)
        optimal_length = calculate_path_length([start, goal], rho=rho)
        summary = read_yips_planning_summary(planning_folder, seq, predictor)
        lens = summary[:, 2]
        times = summary[:, 1]
        ft = np.argwhere(lens > 0.)
        if ft.shape[0] > 0:
            STFPs.append(ft.min())
            TTFP.append(times[ft.min()])
            LOFPs.append(lens[ft.min()])
            LOLPs.append(lens[-1])
            LOnPs.append(lens[ft.min()+100 if ft.min()+100<max_samples else max_samples-1])

            time_threshold = times[ft.min()] + .1 if times[ft.min()] + .1 <= times.max() else times.max()
            lotp = np.argwhere(times <= time_threshold).max()
            LOtPs.append(lens[lotp])

            true_path_lens.append(true_length)
            pred_path_lens.append(pred_length)
            optimal_path_lens.append(optimal_length)
        else:
            Fails.append(seq)

        if ft.shape[0] > 0:
            if times[ft.min()] + infer_times[i][1] < .1:
                SPIT += 1.

        wSTFPs.append(ft.min() if ft.shape[0] else 500)
        wTTFPs.append(times[ft.min()] if ft.shape[0] else 10.)
        wLOFPs.append(lens[ft.min()] if ft.shape[0] else 0.)
        wLOLPs.append(lens[-1] if ft.shape[0] else 0.)
        wLOnPs.append(lens[ft.min()+100 if ft.min()+100 < max_samples else max_samples-1] if ft.shape[0] else 0.)

        if ft.shape[0]:
            time_threshold = times[ft.min()] + .1 if times[ft.min()] + .1 <= times.max() else times.max()
            lotp = np.argwhere(times <= time_threshold).max()
            wLOtPs.append(lens[lotp])
        else:
            wLOtPs.append(0)

        all_true_path_lens.append(true_length)
        all_pred_path_lens.append(pred_length)
        all_optimal_path_lens.append(optimal_length)

    print('=========================================================')
    print('Planner: {}'.format(predictor))
    print 'SPIT: {}/{}'.format(SPIT, SPIT / len(seqs))
    print('Mean STFP: {}, Max STFS: {}@{}, Min STFS: {}, STFS<100: {}@{}'.format(
        np.array(STFPs).mean(), np.array(STFPs).max(), seqs[np.array(STFPs).argmax()], np.array(STFPs).min(),
        (np.array(STFPs) < 100).sum(), 1.*(np.array(STFPs) < 100).sum()/len(seqs)))
    print('Mean TTFP: {}, Max TTFS: {}@{}, Min TTFS: {}, TTFS<0.1: {}@{}'.format(
        np.array(TTFP).mean(), np.array(TTFP).max(), seqs[np.array(TTFP).argmax()], np.array(TTFP).min(),
        (np.array(TTFP) < 0.1).sum(), 1.*(np.array(TTFP) < 0.1).sum()/len(TTFP)))
    print('Mean LOFP: {}, Mean LOLP: {}, Mean LOnP: {}, Mean LOtP: {}, Mean LOYP: {}, Mean LOOP: {}, Mean LOGP: {}'.format(
        np.array(LOFPs).mean(), np.array(LOLPs).mean(), np.array(LOnPs).mean(), np.array(LOtPs).mean(),
        np.array(pred_path_lens).mean(), np.array(true_path_lens).mean(), np.array(optimal_path_lens).mean()))
    print('SR: {}@{}'.format(1. - 1.*len(Fails)/len(seqs), len(Fails)))
    print('Fails Scenes: {}'.format(Fails))
    print('=========================================================')

    return wSTFPs, wTTFPs, wLOFPs, wLOLPs, wLOnPs, wLOtPs, Fails, all_pred_path_lens, all_true_path_lens, all_optimal_path_lens


def extract_ose_summary(dataset_folder, inputs_filename, planning_folder, predictor='ose', max_samples=500):
    if predictor == 'ose':
        summary = read_ose_summary('predictions/valid')
        inference_times = summary[:, 0]
    seqs = read_seqs(dataset_folder, inputs_filename)
    seqs.sort()
    threshold, rho = 0.5, 5.0
    SPIT = 0.
    STFPs, LOFPs, LOLPs, TTFP, Fails, LOnPs, LOtPs = [], [], [], [], [], [], []
    wSTFPs, wLOFPs, wLOLPs, wTTFPs, wLOnPs, wLOtPs = [], [], [], [], [], []
    true_path_lens, optimal_path_lens = [], []
    all_true_path_lens, all_optimal_path_lens = [], []
    for i, seq in enumerate(seqs):  # enumerate(seqs)
        # print('Evaluate Scene: {} ({} of {})'.format(seq, i + 1, len(seqs)))
        label = read_label(dataset_folder, seq)
        start, goal = read_task(dataset_folder, seq)
        true = build_path(start, label, goal, threshold)
        true_length = calculate_path_length(true, rho=rho)
        optimal_length = calculate_path_length([start, goal], rho=rho)
        summary = read_yips_planning_summary(planning_folder, seq, predictor)
        lens = summary[:, 2]
        times = summary[:, 1]
        ft = np.argwhere(lens > 0.)
        if ft.shape[0] > 0:
            STFPs.append(ft.min())
            TTFP.append(times[ft.min()])
            LOFPs.append(lens[ft.min()])
            LOLPs.append(lens[-1])
            LOnPs.append(lens[ft.min()+100 if ft.min()+100 < max_samples else max_samples-1])

            time_threshold = times[ft.min()] + .1 if times[ft.min()] + .1 <= times.max() else times.max()
            lotp = np.argwhere(times <= time_threshold).max()
            LOtPs.append(lens[lotp])

            true_path_lens.append(true_length)
            optimal_path_lens.append(optimal_length)
        else:
            Fails.append(seq)

        if ft.shape[0] > 0:
            if predictor == 'ose':
                if times[ft.min()] + inference_times[i] < .1:
                    SPIT += 1.
            else:
                if times[ft.min()] < .1:
                    SPIT += 1.

        wSTFPs.append(ft.min() if ft.shape[0] else 500)
        wTTFPs.append(times[ft.min()] if ft.shape[0] else 10)
        wLOFPs.append(lens[ft.min()] if ft.shape[0] else 0.)
        wLOLPs.append(lens[-1] if ft.shape[0] else 0.)
        wLOnPs.append(lens[ft.min()+100 if ft.min()+100<max_samples else max_samples-1] if ft.shape[0] else 0.)

        if ft.shape[0]:
            time_threshold = times[ft.min()] + .1 if times[ft.min()] + .1 <= times.max() else times.max()
            lotp = np.argwhere(times <= time_threshold).max()
            wLOtPs.append(lens[lotp])
        else:
            wLOtPs.append(0)

        all_true_path_lens.append(true_length)
        all_optimal_path_lens.append(optimal_length)
    print('=========================================================')
    print('Planner: {}'.format(predictor))
    print 'SPIT: {}/{}'.format(SPIT, SPIT / len(seqs))
    print('Mean STFP: {}, Max STFS: {}@{}, Min STFS: {}, STFS<100: {}@{}'.format(
        np.array(STFPs).mean(), np.array(STFPs).max(), seqs[np.array(STFPs).argmax()], np.array(STFPs).min(),
        (np.array(STFPs) < 100).sum(), 1.*(np.array(STFPs) < 100).sum()/len(seqs)))
    print('Mean TTFP: {}, Max TTFS: {}@{}, Min TTFS: {}, TTFS<0.1: {}@{}'.format(
        np.array(TTFP).mean(), np.array(TTFP).max(), seqs[np.array(TTFP).argmax()], np.array(TTFP).min(),
        (np.array(TTFP) < 0.1).sum(), 1.*(np.array(TTFP) < 0.1).sum()/len(TTFP)))
    print('Mean LOFP: {}, Mean LOLP: {}, Mean LOnP: {}, Mean LOtP: {}, Mean LOOP: {}, Mean LOGP: {}'.format(
        np.array(LOFPs).mean(), np.array(LOLPs).mean(), np.array(LOnPs).mean(), np.array(LOtPs).mean(),
        np.array(true_path_lens).mean(), np.array(optimal_path_lens).mean()))
    print('SR: {}@{}'.format(1. - 1.*len(Fails)/len(seqs), len(Fails)))
    print('Fails Scenes: {}'.format(Fails))
    print('=========================================================')
    return wSTFPs, wTTFPs, wLOFPs, wLOLPs, wLOnPs, wLOtPs, Fails, all_true_path_lens, all_optimal_path_lens


def plot_sorrt_optimization_on_yips(
        YIPSwSTFPs, YIPSwTTFPs, YIPSwLOFPs, YIPSwLOLPs, YIPSwLOnPs, YIPSwLOtPs, YIPSFails,
        all_pred_path_lens, all_true_path_lens, all_optimal_path_lens, seqs):
    data = zip(np.divide(all_true_path_lens, all_optimal_path_lens),
               np.divide(YIPSwLOLPs, all_optimal_path_lens),
               np.divide(all_pred_path_lens, all_optimal_path_lens),
               np.divide(all_optimal_path_lens, all_optimal_path_lens),
               np.divide(YIPSwLOFPs, all_optimal_path_lens),
               np.divide(YIPSwLOtPs, all_optimal_path_lens))
    data.sort()
    data = np.array(data)
    fontsize = 36
    ax1 = new_figure(y_label='$\\widehat{\\mathrm{LOP}}$', x_label='Scene Number', fontsize=fontsize)
    ax1.set_ylim([-0.05, 2.5])
    ax1.set_xlim([0, 2750])
    ax1.set_yticks(np.arange(0., 2.5, 0.5))
    ax1.set_xticks([0, 1000, 2000])
    ax1.plot(data[:, 0], lw=8, c='r', zorder=100, label='Label')
    ax1.plot(data[:, 1], lw=6, c='b', zorder=50, label='$\\mathrm{LOP}_{\\mathrm{500}}$')
    ax1.plot(data[:, 5], lw=6, c='C1', zorder=40, label='$\\mathrm{LOP}_{\\mathrm{TTFP.1}}$')
    ax1.plot(data[:, 4], lw=6, c='y', zorder=25, label='$\\mathrm{LOP}_{\\mathrm{TTFP}}$')
    # ax1.plot(data[:, 2])
    ax1.plot(data[:, 3], lw=8, c='g', label='Geodesic')

    fail_points = np.argwhere(data[:, 1] == 0)
    ax1.scatter(fail_points, [0.] * fail_points.shape[0], s=400, c='r')
    # gaps = data[:, 0] - data[:, 1]
    # tars = filter(lambda x: 0.1 < x < 1, gaps)
    # print tars
    # index = np.argwhere(gaps == tars[0])[0, 0]
    # print index
    # ax1.scatter(index, data[index, 1], s=300, c='r', zorder=100, marker='x', lw=6)
    ax1.legend(prop={'size': fontsize-4}, loc=2, frameon=True, ncol=3, framealpha=1).set_zorder(200)
    #
    # gaps = np.divide(all_true_path_lens, all_optimal_path_lens) - np.divide(YIPSwLOLPs, all_optimal_path_lens)
    # tars = filter(lambda x: 0.1 < x < 1, gaps)
    # print tars, seqs[np.argwhere(gaps == tars[0])[0, 0]]

    fontsize = 40
    ax2 = new_figure(y_label='', x_label='proportion', fontsize=fontsize)
    ax2.set_xlim(0, 1.)
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    h = np.histogram(YIPSwTTFPs, bins=(0, 0.05, 0.1, 0.5, 1., max(YIPSwTTFPs)))
    xlab = ['(0.00, 0.05]', '(0.05, 0.10]', '(0.10, 0.50]', '(0.50, 1.00]', '(1.00, 9.00]']
    ax2.barh(range(5), 1. * h[0] / len(seqs), tick_label=xlab, height=0.6, color='g')
    print 1. * h[0] / len(seqs)
    for i, v in enumerate(1. * h[0] / len(seqs)):
        ax2.text(v + 0.01, i, '{:.5f}'.format(v), color='k', fontdict={'size': fontsize})

    plt.draw()
    plt.show()


def plot_ose_and_yips_and_gau_comparison_length(
        YIPSwSTFPs, YIPSwTTFPs, YIPSwLOFPs, YIPSwLOLPs, YIPSwLOnPs, YIPSwLOtPs, YIPSFails,
        OSEwSTFPs, OSEwTTFPs, OSEwLOFPs, OSEwLOLPs, OSEwLOnPs, OSEwLOtPs, OSEFails,
        GAUwSTFPs, GAUwTTFPs, GAUwLOFPs, GAUwLOLPs, GAUwLOnPs, GAUwLOtPs, GAUFails,
        all_pred_path_lens, all_true_path_lens, all_optimal_path_lens, seqs):

    data = zip(np.divide(all_true_path_lens, all_optimal_path_lens),
               np.divide(YIPSwLOtPs, all_optimal_path_lens),
               np.divide(all_pred_path_lens, all_optimal_path_lens),
               np.divide(all_optimal_path_lens, all_optimal_path_lens),
               np.divide(GAUwLOtPs, all_optimal_path_lens))
    data.sort()
    data = np.array(data)
    fontsize = 36
    ax1 = new_figure(y_label='$\\widehat{\\mathrm{LOP}_{\\mathrm{TTFP.1}}}$', x_label='Scene Number', fontsize=fontsize)
    ax1.set_ylim([-0.05, 2.7])
    ax1.set_xlim([0, 2750])
    ax1.set_yticks(np.arange(0., 2.7, 0.5))
    ax1.set_xticks([0, 1000, 2000])
    ax1.plot(data[:, 0], lw=8, c='r', zorder=100, label='Label')
    ax1.plot(data[:, 1], lw=6, c='b', zorder=50, label='YIPS+SO-RRT*')
    ax1.plot(data[:, 4], lw=6, c='C1', zorder=25, label='GBS+Bi-RRT*')
    # ax1.plot(data[:, 2])
    ax1.plot(data[:, 3], lw=8, c='g', label='Geodesic', zorder=70)

    fail_points = np.argwhere(data[:, 1] == 0)
    ax1.scatter(fail_points, [0.] * fail_points.shape[0], s=200, c='r')
    fail_points = np.argwhere(data[:, 4] == 0)
    ax1.scatter(fail_points, [0.] * fail_points.shape[0], s=200, c='r')
    ax1.legend(prop={'size': fontsize}, loc=2, frameon=True, ncol=2, framealpha=1).set_zorder(102)
    plt.draw()
    plt.show()


def plot_ose_and_yips_and_gau_comparison_times(
        YIPSwSTFPs, YIPSwTTFPs, YIPSwLOFPs, YIPSwLOLPs, YIPSwLOnPs, YIPSwLOtPs, YIPSFails,
        OSEwSTFPs, OSEwTTFPs, OSEwLOFPs, OSEwLOLPs, OSEwLOnPs, OSEwLOtPs, OSEFails,
        GAUwSTFPs, GAUwTTFPs, GAUwLOFPs, GAUwLOLPs, GAUwLOnPs, GAUwLOtPs, GAUFails,
        all_pred_path_lens, all_true_path_lens, all_optimal_path_lens, seqs,
        yips_inference_time, ose_inference_time):

    fontsize = 50
    # TTFP
    ax1 = new_figure(y_label='$\sqrt[4]{\mathrm{TTFP}}$', x_label='', fontsize=fontsize)
    # data = (np.array(YIPSwTTFPs) + np.arange(yips_inference_time)) - (np.array(OSEwTTFPs) + np.array(ose_inference_time))
    # data = np.array(YIPSwTTFPs) - np.array(OSEwTTFPs)
    data = (np.array(YIPSwTTFPs) + np.arange(yips_inference_time)) - np.array(GAUwTTFPs)
    # data = np.array(YIPSwTTFPs) - np.array(GAUwTTFPs)
    data.sort()
    data = np.array(data)
    data = np.multiply(np.sign(data), np.power(np.abs(data), 1 / 4.))

    # STFP
    # ax1 = new_figure(y_label='STFP', x_label='', fontsize=fontsize)
    # ax1.set_yscale('symlog')
    # data = np.array(YIPSwSTFPs) - np.array(OSEwSTFPs)
    # data.sort()
    # data = np.array(data)

    ax1.fill_between(range(data.shape[0]), data, [0] * data.shape[0], facecolors='C2')
    ax1.plot(data, lw=8, c='r', zorder=50, label='mTTFP$^\\Delta$')
    ax1.hlines(0, 0, 2750, lw=8, color='b', zorder=200, linestyles='dashed')
    ax1.legend(prop={'size': fontsize}, loc=2, frameon=True, ncol=2)
    plt.draw()
    plt.show()


def calculate_performance(predictor, dataset_folder, inputs_filename, prediction_folder, planning_folder):
    # set_plot()
    yips_inference_time = calculate_inference_time(predictor, prediction_folder)
    ose_inference_time = calculate_ose_inference_time(prediction_folder)
    print('YIPS Mean Inference Time: {}, OSE Mean Inference Time: {}'.format(
        yips_inference_time, ose_inference_time))
    seqs = read_seqs(dataset_folder, inputs_filename)
    seqs.sort()
    threshold = 0.5
    rho = 5.0

    (YIPSwSTFPs, YIPSwTTFPs, YIPSwLOFPs, YIPSwLOLPs, YIPSwLOnPs, YIPSwLOtPs, YIPSFails,
     all_pred_path_lens, all_true_path_lens, all_optimal_path_lens) = extract_yips_summary(
            predictor=predictor, dataset_folder='../../DataMaker/dataset', inputs_filename='valid.csv',
            prediction_folder='predictions/valid', planning_folder='planned_paths/valid', max_samples=500)

    plot_sorrt_optimization_on_yips(
        YIPSwSTFPs, YIPSwTTFPs, YIPSwLOFPs, YIPSwLOLPs, YIPSwLOnPs, YIPSwLOtPs, YIPSFails,
        all_pred_path_lens, all_true_path_lens, all_optimal_path_lens, seqs)

    (OSEwSTFPs, OSEwTTFPs, OSEwLOFPs, OSEwLOLPs, OSEwLOnPs, OSEwLOtPs, OSEFails,
     all_true_path_lens, all_optimal_path_lens) = extract_ose_summary(
        dataset_folder='../../DataMaker/dataset', inputs_filename='valid.csv',
        planning_folder='planned_paths/valid')

    (GAUwSTFPs, GAUwTTFPs, GAUwLOFPs, GAUwLOLPs, GAUwLOnPs, GAUwLOtPs, GAUFails,
     all_true_path_lens, all_optimal_path_lens) = extract_ose_summary(
        dataset_folder='../../DataMaker/dataset', inputs_filename='valid.csv',
        planning_folder='planned_paths/valid', predictor='none')
    #
    # plot_ose_and_yips_and_gau_comparison_length(
    #     YIPSwSTFPs, YIPSwTTFPs, YIPSwLOFPs, YIPSwLOLPs, YIPSwLOnPs, YIPSwLOtPs, YIPSFails,
    #     OSEwSTFPs, OSEwTTFPs, OSEwLOFPs, OSEwLOLPs, OSEwLOnPs, OSEwLOtPs, OSEFails,
    #     GAUwSTFPs, GAUwTTFPs, GAUwLOFPs, GAUwLOLPs, GAUwLOnPs, GAUwLOtPs, GAUFails,
    #     all_pred_path_lens, all_true_path_lens, all_optimal_path_lens, seqs)

    # plot_ose_and_yips_and_gau_comparison_times(
    #     YIPSwSTFPs, YIPSwTTFPs, YIPSwLOFPs, YIPSwLOLPs, YIPSwLOnPs, YIPSwLOtPs, YIPSFails,
    #     OSEwSTFPs, OSEwTTFPs, OSEwLOFPs, OSEwLOLPs, OSEwLOnPs, OSEwLOtPs, OSEFails,
    #     GAUwSTFPs, GAUwTTFPs, GAUwLOFPs, GAUwLOLPs, GAUwLOnPs, GAUwLOtPs, GAUFails,
    #     all_pred_path_lens, all_true_path_lens, all_optimal_path_lens, seqs,
    #     yips_inference_time, ose_inference_time)

    Debugger.breaker('')


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    predictors = [
        'rgous-vgg19v1C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine150[]_wp0o0e+00)-checkpoint-150',
        'rgous-vgg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr70_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-200',
        'rgous-vgg19C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine200[])-checkpoint-200',
        'rgous-res50PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr30_steps10[30, 140, 170]_wp0o0e+00)-checkpoint-200',
        'rgous-svg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-150',
        'rgous-svg16v1PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[75, 95, 135]_wp0o0e+00)-checkpoint-200',
        'rgous-vgg19v2C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_steps10[75, 105, 135]_wp0o0e+00)-checkpoint-200']

    # planner = predictors[3]
    # print "Evaluate {}".format(planner)
    # extract_prediction_and_ground_truth(
    #     dataset_folder='../../DataMaker/dataset',
    #     inputs_filename='valid.csv',
    #     predictor=target,
    #     folder='predictions/valid')
    # print('Evaluate Predictor: {}'.format(target))

    for planner in [predictors[-1]]:
        calculate_performance(predictor=planner, dataset_folder='../../DataMaker/dataset',
                              inputs_filename='valid.csv', prediction_folder='predictions/valid',
                              planning_folder='planned_paths/valid')
        # extract_yips_summary(predictor=planner, dataset_folder='../../DataMaker/dataset',
        #                      inputs_filename='valid.csv', prediction_folder='predictions/valid',
        #                      planning_folder='planned_paths/valid')
        # extract_ose_summary(dataset_folder='../../DataMaker/dataset',
        #                     inputs_filename='valid.csv', planning_folder='planned_paths/valid')

    # compare_inference_time(predictor=planner, prediction_folder='predictions/valid')