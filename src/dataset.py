import tensorflow as tf
import numpy as np
from PIL import Image
import os


class Pipeline(object):
    def __init__(self, config, root=None):
        self.root = os.getcwd() + os.sep + 'dataset' if not root else root
        self.train = self.generator(channel='train')
        self.valid = self.generator(channel='valid')
        self.cond = self.generator(channel='cond')
        self.config = config

    def generator(self, channel='train'):
        while True:
            f = open(self.root + os.sep + channel + '.csv')
            fl = list(f)
            f.close()
            # batch setting
            if channel != 'cond':
                batch = int(self.config['Train']['batch'])
            else:
                batch = 1
            # channel setting
            if channel == 'train':
                d_size = int(self.config['Train']['ts_size'])
            elif channel == 'valid':
                d_size = int(self.config['Train']['vs_size'])
            else:
                d_size = int(self.config['Pred']['pd_size'])
            # step setting
            step = int(np.ceil(float(d_size) / float(batch)))
            # yield batches
            for i in range(step):
                indices = (np.array(range(batch)) + i * batch) % d_size
                xs, ys = [], []
                for j in indices:
                    # create numpy arrays of input data
                    # and labels, from each line in the file
                    parts = fl[j].rstrip().split(',')
                    x_path, y_path = parts[0], parts[1]
                    x = Image.open(self.root + x_path)
                    x = x.resize(tuple(self.config['Model']['i_shape'][:2]))
                    xs.append(np.array(x))
                    y = np.loadtxt(self.root + y_path, delimiter=',')
                    y = self.preprocess_y(y)
                    ys.append(y)
                yield np.array(xs), np.array(ys)

    def preprocess_y(self, y):
        """[u, v, x, y , theta, c]"""
        a_ = self.config['Model']['A']
        b_ = self.config['Model']['B']
        s_ = self.config['Model']['S']
        w, h = self.config['Model']['w'], self.config['Model']['h']
        if y.shape[0] > 2:
            p_y = np.zeros((y.shape[0] - 2, a_))
            p_y[:, 0] = (y[1:-1, 0] + w / 2) / w  # sigma_x
            p_y[:, 1] = (y[1:-1, 1] + h / 2) / h  # sigma_y
            p_y[:, 2] = np.arctan2(np.sin(y[1:-1, 2]), np.cos(y[1:-1, 2]))
            p_y[:, 2] = (p_y[:, 2] + np.pi) / (2 * np.pi)  # sigma_theta
            p_y[:, 3] = 1  # object
        else:
            p_y = np.zeros((1, a_))
        y_t = []
        for b in range(b_):
            y_t.append(np.zeros((s_, s_, a_)))
        for i in range(p_y.shape[0]):
            p, y = p_y[i], y_t[i]
            y[0, 0, :3] = p[:3]  # x, y, theta
            y[0, 0, 3] = p[3]  # object
        return np.concatenate(y_t, axis=-1)
