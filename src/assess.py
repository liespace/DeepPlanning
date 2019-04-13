import toolbox
import numpy as np
import tensorflow as tf
import os

dir_parent = os.path.dirname(os.getcwd())
dir_food = dir_parent + '/food'
name_food_gcld = '{}/food/gcld.npz'.format(dir_parent)
name_food_gpld = '{}/food/gpld.npz'.format(dir_parent)

with np.load(name_food_gcld) as data:
    feature = data['deltas']
    a = feature[:, :, 2]
    b = np.max(a, axis=-1)
    c = np.max(b, axis=-1)
    print(c)
    print(np.where(a > 5))
