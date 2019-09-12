import tensorflow as tf
import numpy as np
from PIL import Image
import os


class Pipeline(object):
    def __init__(self, config, root=None):
        self.root = os.getcwd() + os.sep + 'dataset' if not root else root
        self.train = self.generator(channel='train')
        self.valid = self.generator(channel='valid')
        self.config = config

    def generator(self, channel='train'):
        while True:
            f = open(self.root + os.sep + channel + '.csv')
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                parts = line.rstrip().split(',')
                x_path, y_path = parts[0], parts[1]
                x = Image.open(self.root + x_path)
                x = x.resize(tuple(self.config['Model']['i_shape'][:2]))
                x = np.array([np.array(x)])
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
            p_y = np.zeros((y.shape[0] - 2, a_ + c_))
            p_y[:, 0] = (y[1:-1, 0] + w / 2) / w  # sigma_x
            p_y[:, 1] = (y[1:-1, 1] + h / 2) / h  # sigma_y
            p_y[:, 2] = np.arctan2(np.sin(y[1:-1, 2]), np.cos(y[1:-1, 2]))
            p_y[:, 2] = (p_y[:, 2] + np.pi) / (2 * np.pi)  # sigma_theta
            p_y[:, 3] = 1  # object
            p_y[:, 4] = y[1:-1, 3]  # class
        else:
            p_y = np.zeros((1, a_ + c_))
        y_t = []
        for b in range(b_):
            y_t.append(np.zeros((s_, s_, a_ + c_)))
        for i in range(p_y.shape[0]):
            p, y = p_y[i], y_t[i]
            y[0, 0, :3] = p[:3]  # x, y, theta
            y[0, 0, 3] = p[3]  # object
            y[0, 0, 4] = p[4]  # class
            break
        return np.concatenate(y_t, axis=-1)
