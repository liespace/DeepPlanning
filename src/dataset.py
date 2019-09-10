import tensorflow as tf
import numpy as np
import cv2
import os


class Pipeline(object):
    def __init__(self, config, root=None):
        self.root = os.getcwd() + os.sep + 'dataset' if not root else root
        self.train = self.generator(channel='train')
        self.valid = self.generator(channel='valid')
        self.config = config

    def generator(self, channel='train', batch=1):
        while True:
            f = open(self.root + os.sep + channel + '.csv')
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                parts = line.rstrip().split(',')
                x_path, y_path = parts[0], parts[1]
                x = cv2.imread(self.root + x_path)
                x = cv2.resize(x, tuple(self.config['Model']['i_shape'][:2]))
                x = np.array([x])
                y = np.loadtxt(self.root + y_path, delimiter=',')
                y = self.preprocess_y(y)
                y = np.array([y])
                yield x, y
            f.close()

    def preprocess_y(self, y):
        """[u, v, x, y , theta, c]"""
        if y.shape[0] > 2:
            w, h = self.config['Model']['w'], self.config['Model']['h']
            w_s = w / self.config['Model']['o_shape'][0]
            h_s = h / self.config['Model']['o_shape'][1]
            p_y = np.zeros((y.shape[0] - 2, y.shape[1] + 2))
            p_y[:, 0] = np.floor((y[1:-1, 0] + w / 2) / w_s)
            p_y[:, 1] = np.floor((y[1:-1, 1] + h / 2) / h_s)
            p_y[:, -4] = (y[1:-1, -4] + w / 2) / w_s - p_y[:, 0]
            p_y[:, -3] = (y[1:-1, -3] + h / 2) / h_s - p_y[:, 1]
            p_y[:, -2] = np.arctan2(np.sin(y[1:-1, -2]), np.cos(y[1:-1, -2]))
            p_y[:, -2] = (p_y[:, -2] + np.pi) / (2 * np.pi)
            p_y[:, -1] = y[1:-1, -1]
        else:
            p_y = np.array([])
        return p_y

    def my_accuracy(self, y_true, y_pred, **kwargs):
        threshold = [0.5, 0.5, np.radians(3)]
        limit = 0.5

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        threshold = tf.cast(threshold, y_pred.dtype)
        limit = tf.cast(limit, y_pred.dtype)

        x = tf.abs(y_pred - y_true)
        x = tf.cast(x < threshold, y_pred.dtype)
        x = tf.reduce_sum(x, axis=-1)
        x = tf.reduce_sum(x, axis=-1)
        x = tf.cast(x >= limit, tf.float32)
        x = tf.keras.backend.mean(x, axis=-1)
        return x
