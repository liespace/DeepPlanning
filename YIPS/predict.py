#!/usr/bin/env python
import os
import re
import time
import glob
import csv
import json
import numpy as np
import tensorflow as tf
from yips.model import DWModel

dataset_folder = '../DataMaker/dataset'
weights_folder_name = 'weights_of_trained_yips'
predictions_folder = 'predictions_of_trained_yips'


def read_inputs(filepath, config):
    for x_path in filepath:
        x = tf.keras.preprocessing.image.load_img(
            x_path, target_size=tuple(config['Model']['i_shape'][:2]), interpolation='bilinear')
        x = tf.keras.preprocessing.image.img_to_array(x)
        if config['Model']['backbone'] == 'res50':
            x = tf.keras.applications.resnet50.preprocess_input(x)
        yield x


def reduction(p):
    p = p.reshape((5, 4))
    p = 1. / (1. + np.exp(-p))
    p[:, -2] = p[:, -2] * 2 * np.pi - np.pi
    p[:, :2] = p[:, 0:2] * 60 - 30
    return p


def save_predictions(y, x_filepath, y_folder, times):
    os.makedirs(y_folder) if not os.path.isdir(y_folder) else None
    names = [re.sub('\\D', '', f.strip().split(',')[0]) + '_inference.txt' for f in x_filepath]
    map(lambda cp: np.savetxt(y_folder + os.sep + cp[0], cp[1], delimiter=','), zip(names, y))
    print ('saved ' + str(len(y)) + ' predictions to folder ' + y_folder)
    summary = csv.writer(open(y_folder + os.sep + '0summary.csv', mode='w'), delimiter=',')
    [summary.writerow([c[0], c[1], c[2]]) for c in zip(names, times, x_filepath)]


def progress(steps):
    number = 0
    while True:
        number += 1
        yield '{}/{}'.format(number, steps)


def plan(weight_folder, x_filename, checkpoint=200):
    # Read dataset
    # inputs_filepath = dataset_folder + os.sep + x_filename + '.csv'
    weight_filepath = weight_folder + os.sep + 'checkpoint-{}.h5'.format(checkpoint)
    config_filename = weight_folder + os.sep + 'config.json'
    y_folder = predictions_folder + os.sep + x_filename + os.sep + weight_folder.split('/')[-1] + '-checkpoint-{}'.format(checkpoint)
    config = json.loads(open(config_filename).read())
    # x_filepath = [f.rstrip().split(',')[0] for f in list(open(inputs_filepath))]
    x_filepath = ['inputs/4960_encoded.png', 'inputs/3520_encoded.png', 'inputs/8320_encoded.png', 'inputs/13760_encoded.png']
    x_filepath = [dataset_folder + os.sep + f for f in x_filepath]

    def predicting(x, step=progress(len(x_filepath))):
        past = time.time()
        p = model.predict_on_sample(np.array([x]))
        t = time.time() - past
        print('{}: time@{} s'.format(next(step), t))
        return reduction(p), t
    # Buildup, Predicting and Saving
    model = DWModel(config).compile().load_weights(weight_filepath)
    model.predict_on_sample(np.array([np.zeros((480, 480, 3))]))  # warm up
    y, times = zip(*map(predicting, read_inputs(x_filepath, config)))
    save_predictions(y, x_filepath, y_folder, times)


def find_files(filepath, form='rgous*'):
    return glob.glob(filepath + os.sep + form)


if __name__ == '__main__':
    # filenames = find_files(weights_folder_name)
    # print([filename.split('/')[-1] for filename in filenames])
    filenames = [
        'rgous-vgg19v1C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine150[]_wp0o0e+00)',
        'rgous-vgg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr70_steps10[70, 95, 110]_wp0o0e+00)',
        'rgous-vgg19C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_cosine200[])',
        'rgous-res50PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr30_steps10[30, 140, 170]_wp0o0e+00)',
        'rgous-svg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[70, 95, 110]_wp0o0e+00)',
        'rgous-vgg19v2C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_steps10[75, 105, 135]_wp0o0e+00)',
        'rgous-svg16v1PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[75, 95, 135]_wp0o0e+00)']
    checkpoints = [150, 200, 200, 200, 150, 200, 200]
    # for i, fn in enumerate([filenames[-3]]):
    #     filename = weights_folder_name + os.sep + fn
    #     print('Run Prediction on Weight: {}'.format(filename))
    #     tf.keras.backend.clear_session()
    #     plan(filename, 'task_fusion', checkpoint=checkpoints[i])

    filename = weights_folder_name + os.sep + filenames[-3]
    plan(filename, 'task_fusion', checkpoint=checkpoints[-3])

