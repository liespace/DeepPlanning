import numpy as np
import os
import tensorflow as tf


class Pipeline(object):
    def __init__(self, config):
        self.train = self.generator(channel='train')
        self.valid = self.generator(channel='valid')
        self.config = config
        self.dataset_folder = config['Train']['dataset']
        self.backbone_name = config['Model']['backbone']
        self.preprocessor = self.get_preprocessor()

    def get_preprocessor(self):
        if self.backbone_name == 'dark53':
            tf.logging.warning('Processing DWDark53')
            return tf.keras.applications.resnet50.preprocess_input
        elif self.backbone_name == 'res50':
            tf.logging.warning('Processing DWRes50')
            return tf.keras.applications.resnet50.preprocess_input
        elif self.backbone_name == 'vgg19':
            tf.logging.warning('Processing DWVGG19')
            return tf.keras.applications.vgg19.preprocess_input
        elif self.backbone_name == 'vgg16':
            tf.logging.warning('Processing DWVGG16')
            return tf.keras.applications.vgg16.preprocess_input
        elif self.backbone_name == 'xception':
            tf.logging.warning('Processing Xception')
            return tf.keras.applications.xception.preprocess_input
        elif self.backbone_name == 'svg16':
            tf.logging.warning('Processing SVG16')
            return tf.keras.applications.vgg16.preprocess_input

    def generator(self, channel='train'):
        while True:
            f = open(self.dataset_folder + os.sep + channel + '.csv')
            fl = list(f)
            f.close()
            # batch setting
            batch = int(self.config['Train']['batch'])
            # channel setting
            if channel == 'train':
                d_size = int(self.config['Train']['training_set_size'])
            else:
                d_size = int(self.config['Train']['validation_set_size'])
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
                    x = self.preprocess_x(x_path)
                    xs.append(x)
                    y = np.loadtxt(self.dataset_folder + os.sep + y_path, delimiter=',')
                    y = self.preprocess_y(y)
                    ys.append(y)
                yield np.array(xs), np.array(ys)

    def preprocess_x(self, x_path):
        x_path = self.dataset_folder + os.sep + x_path
        x = tf.keras.preprocessing.image.load_img(x_path, target_size=tuple(self.config['Model']['i_shape'][:2]), interpolation='bilinear')
        x = tf.keras.preprocessing.image.img_to_array(x)
        # x = self.preprocessor(x)
        return x

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
