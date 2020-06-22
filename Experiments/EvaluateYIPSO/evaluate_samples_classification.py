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
from matplotlib.ticker import FormatStrFormatter


def transform(poly, pto):
    pts = poly.transpose()
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return (np.dot(rot, pts) + xyo).transpose()


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
    return cv2.imread('{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


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
    inferences = np.loadtxt('yips_evaluation/inferences.txt', delimiter=',')
    labels = np.loadtxt('yips_evaluation/labels.txt', delimiter=',')
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
    np.savetxt('yips_evaluation/inferences-{}.txt'.format(predictor), inferences, delimiter=',')
    np.savetxt('yips_evaluation/labels.txt', labels, delimiter=',')


def new_figure(y_label='Precision[-]', x_label='Recall[-]', fontsize=55):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    return ax


def calculate_pr_and_ap(predictor):
    inferences = np.loadtxt('yips_evaluation/inferences-{}.txt'.format(predictor), delimiter=',')
    labels = np.loadtxt('yips_evaluation/labels.txt', delimiter=',')
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

    specific = np.abs(thresholds - 0.71).argmin()
    print(precision[specific], recall[specific], thresholds[specific])
    tri_specific = np.abs(tri_thresholds - 0.71).argmin()
    print(tri_precision[tri_specific], tri_recall[tri_specific], tri_thresholds[tri_specific])

    balance = np.abs(precision - recall).argmin()
    print(balance, precision[balance], recall[balance], thresholds[balance])
    tri_balance = np.abs(tri_precision - tri_recall).argmin()
    print(tri_balance, tri_precision[tri_balance], tri_recall[tri_balance], tri_thresholds[tri_balance])

    fontsize = 36
    ax = new_figure(fontsize=fontsize)
    ax.set_ylim([0., 1.01])
    ax.set_xlim([0., 1.01])
    ax.set_yticks(np.arange(0., 1.01, 0.2))
    ax.set_xticks(np.arange(0., 1.01, 0.2))
    ax.xaxis.get_major_ticks()[0].set_visible(False)
    ax.xaxis.get_major_ticks()[3].set_visible(False)
    ax.xaxis.get_major_ticks()[4].set_visible(False)
    ax.xaxis.get_major_ticks()[5].set_visible(False)
    ax.yaxis.get_major_ticks()[3].set_visible(False)
    ax.yaxis.get_major_ticks()[4].set_visible(False)
    ax.yaxis.get_major_ticks()[5].set_visible(False)
    pr, = ax.plot(recall, precision, linewidth=6)
    tri_pr, = ax.plot(tri_recall, tri_precision, linewidth=6)

    ax.scatter([recall[balance]], [precision[balance]], s=300, color='C3', zorder=100)
    ax.scatter([tri_recall[tri_balance]], [tri_precision[tri_balance]], s=300, color='C3', zorder=100)
    ax.vlines(recall[balance], -0.01, precision[balance], linestyles='dashed', linewidth=4, color='C3', zorder=100)
    ax.hlines(precision[balance], -0.01, recall[balance], linestyles='dashed', linewidth=4, color='C3', zorder=100)
    ax.vlines(tri_recall[tri_balance], -0.01, tri_precision[tri_balance], linestyles='dashed', linewidth=4, color='C3', zorder=100)
    ax.hlines(tri_precision[tri_balance], -0.01, tri_recall[tri_balance], linestyles='dashed', linewidth=4, color='C3', zorder=100)
    plt.xticks(list(plt.xticks()[0]) + [recall[balance], tri_recall[tri_balance]])
    plt.yticks(list(plt.yticks()[0]) + [precision[balance], tri_precision[tri_balance]])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax.annotate('[%.2f]' % thresholds[balance],
                (recall[balance], precision[balance]), fontsize=fontsize - 6)
    ax.annotate('[%.2f]'%tri_thresholds[tri_balance],
                (tri_recall[tri_balance], tri_precision[tri_balance]), fontsize=fontsize-6)

    ax.legend([pr, tri_pr], ['$AP=97.0$', '$AP_{WFS}=78.5$'], prop={'size': fontsize}, frameon=False)
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def calculate_length_pr_and_ap(predictor, dataset_folder, inputs_filename):
    inferences = np.loadtxt('yips_evaluation/inferences-{}.txt'.format(predictor), delimiter=',')
    labels = np.loadtxt('yips_evaluation/labels.txt', delimiter=',')
    ground_truth = labels[:, -1]
    prediction = inferences[:, -1]
    ious = calculate_iou_with_obstacle(inferences, dataset_folder, inputs_filename)
    ious = np.array(ious)
    ious = (ious <= 0.125 * 7).astype(np.float)
    prediction = np.multiply(prediction, ious)
    thresholds = np.arange(0., 1., 1e-3)
    len_true = np.split(ground_truth, len(ground_truth) / 5)
    len_true = np.sum(len_true, -1)
    print np.unique(len_true)
    for i in np.unique(len_true):
        print('Len: {}, Number: {}/{} = {}'.format(
            i, np.sum(len_true == i), len_true.shape[0], 1. * np.sum(len_true == i) / len_true.shape[0]))
    accuracies, accuracies_cla = [], [[], [], [], []]
    for threshold in thresholds:
        thr_pred = (prediction > threshold).astype(np.float)
        len_pred = np.split(thr_pred, len(thr_pred) / 5)
        len_pred = np.sum(len_pred, -1)
        accuracy = np.sum((len_pred == len_true).astype(np.float)) / (len(ground_truth) / 5)
        accuracies.append(accuracy)
        for i, a in enumerate(accuracies_cla):
            len_true_cla = len_true == i + 1
            len_pred_cla = len_pred == i + 1
            accuracy_cla = np.sum((len_pred_cla * len_true_cla).astype(np.float)) / np.sum(len_true_cla)
            a.append(accuracy_cla)
    max_accuracy = max(accuracies)
    best_threshold = thresholds[accuracies.index(max(accuracies))]
    for i, a in enumerate(accuracies_cla):
        print 'Len {} Max Accuracy: {} / {}'.format(i+1, max(a), thresholds[a.index(max(a))])
        # plt.plot(thresholds, a)
    print 'Len Max Accuracy: {} / {}'.format(max_accuracy, best_threshold)
    for i, a in enumerate(accuracies_cla):
        print '  Len {} Accuracy: {}'.format(i+1, a[accuracies.index(max(accuracies))])

    fontsize = 36
    ax = new_figure(y_label='Accuracy[-]', x_label='Threshold[-]', fontsize=fontsize)
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlim([-0.01, 1.01])
    ax.set_yticks(np.arange(0., 1.01, 0.2))
    ax.set_xticks(np.arange(0., 1.01, 0.2))
    ax.xaxis.get_major_ticks()[0].set_visible(False)
    acc, = ax.plot(thresholds, accuracies, linewidth=6, color='C3', zorder=50)
    acc1, = ax.plot(thresholds, accuracies_cla[0], linewidth=4, color='C0')
    acc2, = ax.plot(thresholds, accuracies_cla[1], linewidth=4, color='C1')
    acc3, = ax.plot(thresholds, accuracies_cla[2], linewidth=4, color='C2')
    acc4, = ax.plot(thresholds, accuracies_cla[3], linewidth=4, color='C4')

    ax.legend([acc, acc1, acc2, acc3, acc4],
              # ['$Acc$', '$Acc^1$', '$Acc^2$', '$Acc^3$', '$Acc^4$']
              ['$Acc_{IOV.0}$', '$Acc^1_{IOV.0}$', '$Acc^2_{IOV.0}$', '$Acc^3_{IOV.0}$', '$Acc^4_{IOV.0}$'],
              prop={'size': fontsize}, frameon=False, ncol=2,
              #  loc='lower right', bbox_to_anchor=(0.77, 0.71)
              loc=2)
    ax.vlines(best_threshold, -0.01, max_accuracy, linestyles='dashed', linewidth=3, color='C9', zorder=100)
    ax.hlines(max_accuracy, -0.01, best_threshold, linestyles='dashed', linewidth=3, color='C9', zorder=100)
    plt.xticks(list(plt.xticks()[0]) + [best_threshold])
    plt.yticks(list(plt.yticks()[0]) + [max_accuracy])
    ax.scatter([best_threshold], [max_accuracy], s=200, color='C9', zorder=100)
    ax.xaxis.get_major_ticks()[2].set_visible(False)
    ax.xaxis.get_major_ticks()[3].set_visible(False)
    ax.yaxis.get_major_ticks()[2].set_visible(False)
    ax.yaxis.get_major_ticks()[3].set_visible(False)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def calculate_free_pr_and_ap(predictor, dataset_folder, inputs_filename):
    inferences = np.loadtxt('yips_evaluation/inferences-{}.txt'.format(predictor), delimiter=',')
    labels = np.loadtxt('yips_evaluation/labels.txt', delimiter=',')
    ground_truth = list(labels[:, -1])
    prediction = list(inferences[:, -1])
    ious = calculate_iou_with_obstacle(inferences, dataset_folder, inputs_filename)
    ious = np.array(ious)
    ious = (ious <= 0.125*0).astype(np.float)

    pro_prediction = np.multiply(np.array(prediction), ious)
    average_precision = average_precision_score(ground_truth, pro_prediction)
    print('AP_Free: {}'.format(average_precision))

    tri_ground_truth, tri_prediction = trimmed_prediction_and_ground_truth(ground_truth, prediction)
    tri_ious = trimmed_ious(ious)
    tri_pro_prediction = np.multiply(np.array(tri_prediction), tri_ious)
    tri_average_precision = average_precision_score(tri_ground_truth, tri_pro_prediction)
    print('Trimmed AP_Free: {}'.format(tri_average_precision))

    precision_ioc, recall_ioc, thresholds_ioc = precision_recall_curve(ground_truth, pro_prediction)
    tri_precision_ioc, tri_recall_ioc, tri_thresholds_ioc = precision_recall_curve(tri_ground_truth, tri_pro_prediction)

    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction)
    tri_precision, tri_recall, tri_thresholds = precision_recall_curve(tri_ground_truth, tri_prediction)

    balance = np.abs(precision - recall).argmin()
    print(balance, precision[balance], recall[balance], thresholds[balance])
    tri_balance = np.abs(tri_precision - tri_recall).argmin()
    print(tri_balance, tri_precision[tri_balance], tri_recall[tri_balance], tri_thresholds[tri_balance])
    balance_ioc = np.abs(precision_ioc - recall_ioc).argmin()
    print(balance_ioc, precision_ioc[balance_ioc], recall_ioc[balance_ioc], thresholds_ioc[balance_ioc])
    tri_balance_ioc = np.abs(tri_precision_ioc - tri_recall_ioc).argmin()
    print(tri_balance_ioc, tri_precision_ioc[tri_balance_ioc], tri_recall_ioc[tri_balance_ioc], tri_thresholds_ioc[tri_balance_ioc])

    fontsize = 36
    ax = new_figure(fontsize=fontsize)
    ax.set_ylim([0., 1.01])
    ax.set_xlim([0., 1.01])
    ax.set_yticks(np.arange(0., 1.01, 0.2))
    ax.set_xticks(np.arange(0., 1.01, 0.2))
    ax.xaxis.get_major_ticks()[0].set_visible(False)
    pr, = ax.plot(recall, precision, linewidth=6)
    tri_pr, = ax.plot(tri_recall, tri_precision, linewidth=6)
    pr_ioc, = ax.plot(recall_ioc[1:], precision_ioc[1:], linewidth=6)
    tri_pr_ioc, = ax.plot(tri_recall_ioc[1:], tri_precision_ioc[1:], linewidth=6)
    ax.legend([pr, pr_ioc, tri_pr, tri_pr_ioc],
              ['$AP=97.0$', '$AP_{IOV.0}=75.1$', '$AP_{WFS}=78.5$', '$AP_{WFS+IOV.0}=55.5$'],
              prop={'size': fontsize}, loc=3, frameon=False)

    ax.scatter([recall[balance]], [precision[balance]], s=300, color='C3', zorder=100)
    ax.scatter([tri_recall[tri_balance]], [tri_precision[tri_balance]], s=300, color='C3', zorder=100)
    ax.scatter([recall_ioc[balance_ioc]], [precision_ioc[balance_ioc]], s=300, color='C3', zorder=100)
    ax.scatter([tri_recall_ioc[tri_balance_ioc]], [tri_precision_ioc[tri_balance_ioc]], s=300, color='C3', zorder=100)
    ax.annotate('[%.2f]' % thresholds[balance],
                (recall[balance], precision[balance]), fontsize=fontsize - 6)
    ax.annotate('[%.2f]' % tri_thresholds[tri_balance],
                (tri_recall[tri_balance], tri_precision[tri_balance]), fontsize=fontsize - 6)
    ax.annotate('[%.2f]' % thresholds_ioc[balance_ioc],
                (recall_ioc[balance_ioc], precision_ioc[balance_ioc]), fontsize=fontsize - 6)
    ax.annotate('[%.2f]' % tri_thresholds_ioc[tri_balance_ioc],
                (tri_recall_ioc[tri_balance_ioc], tri_precision_ioc[tri_balance_ioc]), fontsize=fontsize - 6)

    ax.xaxis.get_major_ticks()[5].set_visible(False)
    ax.yaxis.get_major_ticks()[5].set_visible(False)

    ax.set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    predictors = [
        'rgous-vgg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr70_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-200',
        'rgous-res50PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr30_steps10[30, 140, 170]_wp0o0e+00)-checkpoint-200',
        'rgous-svg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-150',
        'rgous-vgg19v2C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_steps10[75, 105, 135]_wp0o0e+00)-checkpoint-200']

    yips = predictors[-1]
    print "Evaluate {}".format(yips)
    # extract_prediction_and_ground_truth(
    #     dataset_folder='../../DataMaker/dataset',
    #     inputs_filename='valid.csv',
    #     predictor=yips,
    #     folder='predictions/valid')
    calculate_pr_and_ap(predictor=yips)
    calculate_free_pr_and_ap(predictor=yips, dataset_folder='../../DataMaker/dataset', inputs_filename='valid.csv')
    calculate_length_pr_and_ap(predictor=yips, dataset_folder='../../DataMaker/dataset', inputs_filename='valid.csv')
