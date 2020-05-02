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
import scipy.stats as st


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
    np.savetxt('inferences-{}.txt'.format(predictor), inferences, delimiter=',')
    np.savetxt('labels.txt', labels, delimiter=',')


def new_figure(y_label='Precision[-]', x_label='Recall[-]', fontsize=55):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    return ax


def calculate_pdf_and_cdf(predictor):
    inferences = np.loadtxt('inferences-{}.txt'.format(predictor), delimiter=',')
    labels = np.loadtxt('labels.txt', delimiter=',')
    errors = []
    print(inferences.shape)
    for i in range(inferences.shape[0]):
        infer, label = inferences[i], labels[i]
        if label[-1] == 1:
            error = label[:-1] - infer[:-1]
            error[-1] = (error[-1] + np.pi) % (2 * np.pi) - np.pi
            errors.append(list(error))
    errors = np.array(errors)
    print(errors[:, 0])

    fontsize = 70
    error_number = 2
    x_e = errors[:, error_number]
    error_labels = ['$X_e$', '$Y_e$', '$\\Phi_e$']
    x_es = np.linspace(errors[:, error_number].min(), errors[:, error_number].max(), 300)
    bins = 40
    print('Skewness: {}, kurtosis: {}, K2&P-value: {}'.format(
        st.skew(x_e), st.kurtosis(x_e), st.normaltest(x_e)))
    # ax = new_figure(fontsize=fontsize, y_label='Probability', x_label=error_labels[error_number])
    # ax.hist(x_e, bins=bins, density=1, histtype='bar', facecolor='C1', alpha=1.0,
    #         cumulative=True, rwidth=0.8, linewidth=12, color='C1', label='Data')
    # ax.plot(x_es, st.rv_histogram(np.histogram(x_e, bins=bins)).cdf(x_es), linewidth=12, color='C0', label='CDF')
    # ax.legend(prop={'size': fontsize}, loc=2)

    ax1 = new_figure(fontsize=fontsize, y_label='', x_label=error_labels[error_number])
    ax1.hist(x_e, bins=bins, density=1, histtype='bar', facecolor='C1',
             alpha=1.0, cumulative=False, rwidth=0.8, linewidth=12, color='C1', label='Data')
    ax1.plot(x_es, st.gaussian_kde(x_e).pdf(x_es), linewidth=12, color='C0', label='PDF')
    ax1.plot(x_es, st.rv_histogram(np.histogram(x_e, bins=bins)).cdf(x_es), linewidth=12, color='C3', label='CDF')
    ax1.legend(prop={'size': fontsize}, loc=2, frameon=False)
    plt.show()


def calculate_pdf_and_cdf2(targets):
    fontsize = 70
    bins = 40
    error_number = 1
    error_labels = ['$X_e$', '$Y_e$', '$\\Phi_e$']
    plot_labels = ['VGG-19', 'SVG-16', 'VGG-16', 'ResNet-50']
    plot_labels.reverse()
    ax = new_figure(fontsize=fontsize, y_label='', x_label=error_labels[error_number])
    ax1 = new_figure(fontsize=fontsize, y_label='', x_label=error_labels[error_number])
    targets.reverse()
    print targets
    for j, tar in enumerate(targets):
        inferences = np.loadtxt('inferences-{}.txt'.format(tar), delimiter=',')
        labels = np.loadtxt('labels.txt', delimiter=',')
        errors = []
        for i in range(inferences.shape[0]):
            infer, label = inferences[i], labels[i]
            if label[-1] == 1:
                error = label[:-1] - infer[:-1]
                error[-1] = (error[-1] + np.pi) % (2 * np.pi) - np.pi
                errors.append(list(error))
        errors = np.array(errors)

        x_e = errors[:, error_number]
        x_es = np.linspace(errors[:, error_number].min(), errors[:, error_number].max(), 300)
        # ax.hist(x_e, bins=bins, density=1, histtype='bar', facecolor='C1', alpha=1.0,
        #         cumulative=True, rwidth=0.8, linewidth=12, color='C1', label='Data')
        ax.plot(x_es, st.rv_histogram(np.histogram(x_e, bins=bins)).cdf(x_es), linewidth=12, color='C{}'.format(j), label=plot_labels[j])
        ax.legend(prop={'size': fontsize}, loc=2, frameon=False)

        # ax1.hist(x_e, bins=bins, density=1, histtype='bar', facecolor='C1',
        #          alpha=1.0, cumulative=False, rwidth=0.8, linewidth=12, color='C1', label='Data')
        ax1.plot(x_es, st.gaussian_kde(x_e).pdf(x_es), linewidth=12, color='C{}'.format(j), label=plot_labels[j])
        ax1.legend(prop={'size': fontsize}, loc=2, frameon=False)
    plt.show()


def calculate_pp_plot(predictor):
    inferences = np.loadtxt('inferences-{}.txt'.format(predictor), delimiter=',')
    labels = np.loadtxt('labels.txt', delimiter=',')
    errors = []
    for i in range(inferences.shape[0]):
        infer, label = inferences[i], labels[i]
        if label[-1] == 1:
            error = label[:-1] - infer[:-1]
            error[-1] = (error[-1] + np.pi) % (2 * np.pi) - np.pi
            errors.append(list(error))
    errors = np.array(errors)

    fontsize = 70
    error_number = 2
    x_e = errors[:, error_number]
    error_labels = ['$X_e$', '$Y_e$', '$\\Phi_e$']
    ax = new_figure(fontsize=fontsize, y_label='', x_label='')
    res = st.probplot(x_e, plot=None)
    ax.set_title('')
    ax.set_xticks([-2, 0, 2])
    k, b = res[-1][0], res[-1][1]
    print 'Sigma: {}, Mu: {}'.format(k, b)
    ax.scatter(res[0][0], res[0][1], s=500, c='b', zorder=100, label='Data')
    ax.plot([-3.6, 3.6], [-3.6 * k + b, k * 3.6 + b], linewidth=12, zorder=1000, color='r', label='Best-fit')
    ax.legend(prop={'size': fontsize}, loc=2)
    # ax.set_xticklabels([abs(x) for x in ax.get_xticks()])
    plt.show()


def calculate_pp_plot_without_free_large_error(predictor, dataset_folder, inputs_filename):
    inferences = np.loadtxt('inferences-{}.txt'.format(predictor), delimiter=',')
    labels = np.loadtxt('labels.txt', delimiter=',')
    ious = calculate_iou_with_obstacle(inferences, dataset_folder, inputs_filename)
    ious = np.array(ious)
    ious = (ious <= 0.125 * 0)
    errors = []
    for i in range(inferences.shape[0]):
        if ious[i]:
            continue
        infer, label = inferences[i], labels[i]
        if label[-1] == 1:
            error = label[:-1] - infer[:-1]
            error[-1] = (error[-1] + np.pi) % (2 * np.pi) - np.pi
            errors.append(list(error))
    errors = np.array(errors)

    fontsize = 70
    error_number = 0
    x_e = errors[:, error_number]
    error_labels = ['$X_e$', '$Y_e$', '$\\Phi_e$']
    ax = new_figure(fontsize=fontsize, y_label=error_labels[error_number], x_label='')
    res = st.probplot(x_e, plot=None)
    ax.set_title('')
    ax.set_xlabel('', fontsize=fontsize)
    ax.set_ylabel(error_labels[error_number], fontsize=fontsize)
    ax.set_xticks([-2, 0, 2])
    k, b = res[-1][0], res[-1][1]
    print 'Sigma: {}, Mu: {}'.format(k, b)

    # errors_t = []
    # for i in range(inferences.shape[0]):
    #     infer, label = inferences[i], labels[i]
    #     if label[-1] == 1:
    #         error_t = label[:-1] - infer[:-1]
    #         if error_t[error_number] > 2*k and ious[i]:
    #             continue
    #         error_t[-1] = (error_t[-1] + np.pi) % (2 * np.pi) - np.pi
    #         errors_t.append(list(error_t))
    # errors_t = np.array(errors_t)

    ax.scatter(res[0][0], res[0][1], s=500, c='b', zorder=100, label='Data')
    ax.plot([-3.6, 3.6], [-3.6 * k + b, k * 3.6 + b], linewidth=12, zorder=1000, color='r', label='Best-fit')
    ax.legend(prop={'size': fontsize}, loc=2)
    # ax.set_xticklabels([abs(x) for x in ax.get_xticks()])
    plt.show()


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    predictors = [
        #'rgous-vgg19v1C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine150[]_wp0o0e+00)-checkpoint-150',
        #'rgous-vgg19C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine200[])-checkpoint-200',
        #'rgous-svg16v1PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[75, 95, 135]_wp0o0e+00)-checkpoint-200',
        'rgous-vgg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr70_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-200',
        'rgous-res50PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr30_steps10[30, 140, 170]_wp0o0e+00)-checkpoint-200',
        'rgous-svg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[70, 95, 110]_wp0o0e+00)-checkpoint-150',
        'rgous-vgg19v2C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_steps10[75, 105, 135]_wp0o0e+00)-checkpoint-200'
    ]

    target = predictors[-1]
    print "Evaluate {}".format(target)
    # extract_prediction_and_ground_truth(
    #     dataset_folder='../../DataMaker/dataset',
    #     inputs_filename='valid.csv',
    #     predictor=target,
    #     folder='predictions/valid')
    # print('Evaluate Predictor: {}'.format(target))
    # calculate_pdf_and_cdf(predictor=target)
    # calculate_pdf_and_cdf2(targets=predictors)
    calculate_pp_plot(target)
    # calculate_pp_plot_without_free_large_error(target, dataset_folder='../../DataMaker/dataset', inputs_filename='valid.csv')
