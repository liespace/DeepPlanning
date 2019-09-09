import tensorflow as tf
import numpy as np
import cv2
import os


class Pipeline(object):
    def __init__(self, root=None):
        self.root = os.getcwd() + os.sep + 'dataset' if not root else root
        self.train = self.generator(channel='train')
        self.valid = self.generator(channel='valid')

    def generator(self, channel='train', batch=1):
        while True:
            f = open(self.root + os.sep + channel + '.csv')
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                parts = line.rstrip().split(',')
                x_path, y_path = parts[0], parts[1]
                x = cv2.imread(self.root + x_path)
                x = cv2.resize(x, (480, 480))
                y = np.loadtxt(self.root + y_path, delimiter=',')
                y[:, 2] = np.arctan2(np.sin(y[:, 2]), np.cos(y[:, 2]))
                yield ({'input': x}, {'output': y})
            f.close()

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
