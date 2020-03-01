#!/usr/bin/env python
import os
import re
import time
import sys
import cv2
import json
import numpy as np
sys.path.append('./YIPS')
from yips.model import DWModel

dataset_folder = './Dataset'
weights_folder_name = './YIPS/weights_of_trained_yips'
predictions_folder = './predictions'
config_filename = './YIPS/config.json'
weights_suffix = '.h5'
x_suffix = '.csv'
y_suffix = '_pred.txt'


def read_input(filepath, size=(480, 480)):
    x = cv2.imread(dataset_folder + filepath)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, size)
    return x


def reduction(p):
    p = p.reshape((5, 4))
    p = 1. / (1. + np.exp(-p))
    p[:, -2] = p[:, -2] * 2 * np.pi - np.pi
    p[:, :2] = p[:, 0:2] * 60 - 30
    return p


def main(weight_filename, x_filename):
    # Read dataset
    inputs_filepath = dataset_folder + os.sep + x_filename + x_suffix
    weight_filepath = weights_folder_name + os.sep + weight_filename + weights_suffix
    y_folder = predictions_folder + os.sep + x_filename + os.sep + weight_filename
    config = json.loads(open(config_filename).read())
    x_filepath = [f.rstrip().split(',')[0] for f in list(open(inputs_filepath))]
    model = DWModel(config).compile().load_weights(weight_filepath)
    for f in x_filepath[:10]:
        print('Processing Scene: {}'.format(f))
        x = read_input(f)

        def predicting(i):
            past = time.time()
            model.predict_on_sample(np.array([x]))
            return time.time() - past
        runtime = map(predicting, range(10))

        y, seq = reduction(model.predict_on_sample(np.array([x]))), re.sub('\\D', '', f)
        os.makedirs(y_folder) if not os.path.isdir(y_folder) else None
        np.savetxt('{}/{}_inference.txt'.format(y_folder, seq), y, delimiter=',')
        np.savetxt('{}/{}_summary.txt'.format(y_folder, seq), runtime, delimiter=',')
        print('    Runtime: {}s'.format(np.mean(runtime)))


if __name__ == '__main__':
    main('vgg19_comp_free200_check300', 'test')
