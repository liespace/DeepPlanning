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
import logging
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def transform(poly, pto):
    pts = poly.transpose()
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return (np.dot(rot, pts) + xyo).transpose()


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
        ious.append(np.sum(true_and_pred/255.) / np.sum(true_or_pred/255.))
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


def rectangle(wheelbase=2.850, width=2.116 + 0.2, length=4.925 + 0.2):  # 2.96, 2.2, 5.0
    return np.array([
        [-(length / 2. - wheelbase / 2.), width / 2.], [length / 2. + wheelbase / 2., width / 2.],
        [length / 2. + wheelbase / 2., -width / 2.], [-(length / 2. - wheelbase / 2.), -width / 2.]])


def read_inference(filepath, seq, folder='vgg19_comp_free200_check300', discrimination=0.7):
    inference = np.loadtxt('{}/{}/{}_inference.txt'.format(filepath, folder, seq), delimiter=',')
    # inference = filter(lambda x: x[-1] > discrimination, inference)
    return inference


def read_seqs(dataset_folder, inputs_filename):
    inputs_filepath = dataset_folder + os.sep + inputs_filename
    if 'csv' in inputs_filename:
        file_list = [f.rstrip().split(',')[0] for f in list(open(inputs_filepath))]
    else:
        file_list = os.listdir(inputs_filepath)
    return [re.sub('\\D', '', f.strip().split(',')[0]) for f in file_list]


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


def extract_prediction_and_ground_truth(dataset_folder, inputs_filename, predictor, folder):
    seqs = read_seqs(dataset_folder, inputs_filename)
    seqs.sort()
    prediction, ground_truth = [], []
    inferences, labels = [], []
    for i, seq in enumerate(seqs):  # enumerate(seqs)
        print('Evaluate Scene: {} ({} of {})'.format(seq, i+1, len(seqs)))
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
    np.savetxt('necessity_prediction.txt', prediction, delimiter=',')
    np.savetxt('necessity_ground_truth.txt', ground_truth, delimiter=',')
    np.savetxt('inferences.txt', inferences, delimiter=',')
    np.savetxt('labels.txt', labels, delimiter=',')


def calculate_pr_and_ap():
    inferences = np.loadtxt('inferences.txt', delimiter=',')
    labels = np.loadtxt('labels.txt', delimiter=',')
    ground_truth = list(labels[:, -1])
    prediction = list(inferences[:, -1])
    print sum(ground_truth), len(ground_truth)
    trimmed_ground_truth, trimmed_prediction = trimmed_prediction_and_ground_truth(ground_truth, prediction)
    print('Trimmed NumberOfSamples: {}/{}'.format(sum(trimmed_ground_truth), len(trimmed_ground_truth)))
    average_precision = average_precision_score(ground_truth, prediction)
    trimmed_average_precision = average_precision_score(trimmed_ground_truth, trimmed_prediction)
    print('AP: {}, Trimmed AP: {}'.format(average_precision, trimmed_average_precision))

    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction)
    tri_precision, tri_recall, tri_thresholds = precision_recall_curve(trimmed_ground_truth, trimmed_prediction)
    print(thresholds)
    print(thresholds.shape)
    plt.plot(recall, precision)
    plt.plot(tri_recall, tri_precision)
    # plt.show()


def calculate_length_pr_and_ap():
    inferences = np.loadtxt('inferences.txt', delimiter=',')
    labels = np.loadtxt('labels.txt', delimiter=',')
    ground_truth = labels[:, -1]
    prediction = inferences[:, -1]
    thresholds = np.arange(0., 1., 1e-4)
    len_true = np.split(ground_truth, len(ground_truth)/5)
    len_true = np.sum(len_true, -1)
    print np.unique(len_true)
    for i in np.unique(len_true):
        print('Len: {}, Number: {}/{} = {}'.format(
            i, np.sum(len_true==i), len_true.shape[0], 1.*np.sum(len_true==i)/len_true.shape[0]))
    accuracies, accuracies_cla = [], [[], [], [], []]
    for threshold in thresholds:
        thr_pred = (prediction > threshold).astype(np.float)
        len_pred = np.split(thr_pred, len(thr_pred)/5)
        len_pred = np.sum(len_pred, -1)
        accuracy = np.sum((len_pred == len_true).astype(np.float)) / (len(ground_truth)/5)
        accuracies.append(accuracy)
        for i, a in enumerate(accuracies_cla):
            len_true_cla = len_true == i+1
            len_pred_cla = len_pred == i+1
            accuracy_cla = np.sum((len_pred_cla * len_true_cla).astype(np.float)) / np.sum(len_true_cla)
            a.append(accuracy_cla)
    for a in accuracies_cla:
        print max(a), thresholds[a.index(max(a))]
        plt.plot(thresholds, a)
    max_accuracy = max(accuracies)
    best_threshold = thresholds[accuracies.index(max(accuracies))]
    print max_accuracy, best_threshold
    plt.plot(thresholds, accuracies)
    plt.vlines(best_threshold, 0, max_accuracy)
    plt.hlines(max_accuracy, 0, best_threshold)
    plt.show()


if __name__ == '__main__':
    # 'rgous-vgg19v1C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine150[]_wp0o0e+00)',
    # 'rgous-vgg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr70_steps10[70, 95, 110]_wp0o0e+00)',
    # 'rgous-vgg19C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine200[])',
    # 'rgous-res50PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr30_steps10[30, 140, 170]_wp0o0e+00)',
    # 'rgous-svg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[70, 95, 110]_wp0o0e+00)',
    # 'rgous-vgg19v2C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_steps10[75, 105, 135]_wp0o0e+00)-checkpoint-200'

    # extract_prediction_and_ground_truth(
    #     dataset_folder='../../DataMaker/dataset',
    #     inputs_filename='valid.csv',
    #     predictor='rgous-vgg19v2C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_steps10[75, 105, 135]_wp0o0e+00)-'
    #                'checkpoint-200',
    #     folder='predictions/valid')

    # calculate_pr_and_ap()
    calculate_length_pr_and_ap()
