import toolbox
import numpy as np
import tensorflow as tf
import os
import random

dir_parent = os.path.dirname(os.getcwd())
dir_food = dir_parent + '/food'
name_food_gcld = '{}/food/gcld.npz'.format(dir_parent)
name_food_gpld = '{}/food/gpld.npz'.format(dir_parent)

in2 = tf.keras.layers.Input(shape=(0,), dtype=tf.float32, name='in2')
print(in2.shape)

# with np.load(name_food_gcld) as data:
#     feature = data['deltas']
#     a = feature[:, :, 2]
#     b = np.max(a, axis=-1)
#     c = np.max(b, axis=-1)
#     print(c)
#     print(np.where(a > 5))

# def g():
#     inputs = []
#     outputs = []
#     batch_size = 1
#     menu = {'in': ['gridmap', 'condition'], 'out': ['label']}
#     with np.load('{}/food/{}_{}.{}'.format(dir_parent, 'gpld', 'train', 'npz')) as repo:
#         for key in menu['in']:
#             inputs.append(repo[key])
#         for key in menu['out']:
#             outputs.append(repo[key])
#     indexes = list(range(0, inputs[0].shape[0]))
#     while True:
#         index = random.sample(indexes, k=batch_size)
#         print(index)
#         ins = []
#         for i, key in enumerate(menu['in']):
#             value = []
#             for j in index:
#                 x = inputs[i][j]
#                 if key == 'gridmap':
#                     x /= 255
#                 value.append(x)
#             ins.append((key, np.array(value)))
#         outs = []
#         for i, key in enumerate(menu['out']):
#             value = []
#             for j in index[0:batch_size]:
#                 x = outputs[i][j]
#                 if key == 'gridmap':
#                     x /= 255
#                 value.append(x)
#             outs.append((key, np.array(value)))
#         yield (dict(ins), dict(outs))
#
#
# f = g()
# while True:
#     print(next(f))
#     input('hello')
