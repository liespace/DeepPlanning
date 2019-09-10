from __future__ import print_function
import numpy as np
import cv2
import os
import tensorflow as tf
from cores import DWDark53, DWRes50, DWVGG19, DWDark19

# print (os.getcwd() + os.sep + 'dataset')
# root = './dataset'
# f = open(root + '/train' + '.csv')
# for line in f:
#     # create numpy arrays of input data
#     # and labels, from each line in the file
#     parts = line.rstrip().split(',')
#     print (parts[0])
#     x_path, y_path = str(parts[0]), str(parts[1])
#     print (x_path, y_path)
#     x = cv2.imread(root + x_path)
#     y = np.loadtxt(root + y_path, delimiter=',')
#     y[:, 2] = np.arctan2(np.sin(y[:, 2]), np.cos(y[:, 2]))
#     print (y)
#     print (x.shape)
#     break
# f.close()

# self.ipu = tf.keras.applications.ResNet50(input_shape=self.input_shape,
#                                                   weights=None,
#                                                   include_top=False)
# self.ipu.load_weights(self.ipu_weight)
# self.oru = tf.keras.Sequential()
# self.oru.add(tf.keras.layers.Reshape(target_shape=self.output_shape,
#                                      input_shape=(
#                                      np.prod(self.output_shape),)))


# fen = tf.keras.applications.VGG19(
#     include_top=False,
#     input_shape=(480, 480, 3),
#     weights=None)
# model = tf.keras.Model(inputs=fen.input, outputs=fen.output)
# model.summary()

# model = DWDark53((480, 480, 3), 3, 1, 4)
# model.summary()

# model = DWRes50((480, 480, 3), 3, 1, 4)
# model.summary()

# model = DWVGG19((480, 480, 3), 3, 1, 4)
# model.summary()

# model = DWDark19((480, 480, 3), 3, 1, 4)
# model.summary()

uv = tf.keras.backend.variable([[1, 2, 3], [1, 2, 3]])
uv = uv[:, 0:2]
wh = tf.where(tf.equal(uv, 3))
ewh = tf.keras.backend.eval(wh)
b = tf.keras.backend.eval(uv[:, 0])
print(b, type(b))
print(uv[:, 0].shape)
