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
        a_ = self.config['Model']['A']
        b_ = self.config['Model']['B']
        c_ = self.config['Model']['C']
        s_ = self.config['Model']['S']
        w, h = self.config['Model']['w'], self.config['Model']['h']
        if y.shape[0] > 2:
            w_s, h_s = w / s_, h / s_
            p_y = np.zeros((y.shape[0] - 2, y.shape[1] + 2 + 1))
            p_y[:, 0] = np.floor((y[1:-1, 0] + w / 2) / w_s)  # u
            p_y[:, 1] = np.floor((y[1:-1, 1] + h / 2) / h_s)  # v
            p_y[:, 2] = 1  # obj
            p_y[:, -4] = (y[1:-1, -4] + w / 2) / w_s - p_y[:, 0]  # sigma_x
            p_y[:, -3] = (y[1:-1, -3] + h / 2) / h_s - p_y[:, 1]  # sigma_y
            p_y[:, -2] = np.arctan2(np.sin(y[1:-1, -2]), np.cos(y[1:-1, -2]))
            p_y[:, -2] = (p_y[:, -2] + np.pi) / (2 * np.pi)    # sigma_theta
            p_y[:, -1] = y[1:-1, -1]  # class
        else:
            p_y = np.zeros((1, 2 + a_ + c_))
        y_t = []
        for b in range(b_):
            y_t.append(np.zeros((s_, s_, a_ + c_)))
        for i in range(p_y.shape[0]):
            p = p_y[i]
            for y in y_t:
                if not y[int(p[0]), int(p[1]), a_-1]:
                    y[int(p[0]), int(p[1]), :a_-1] = p[-a_: -c_]  # x, y, theta
                    y[int(p[0]), int(p[1]), a_-1] = p[-c_-a_]  # obj
                    y[int(p[0]), int(p[1]), a_:] = p[-c_:]  # class
                    break
        return np.concatenate(y_t, axis=-1)
