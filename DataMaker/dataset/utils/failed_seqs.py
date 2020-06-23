#! /usr/bin/env python
import os
from shutil import move, copy
import numpy as np


label_folder = 'labels'
os.makedirs(label_folder) if not os.path.isdir(label_folder) else None
fileList = os.listdir('inputs')

amount = 0
for name in fileList:
    seq = name.split('_')[0]
    label_length_filepath = label_folder + os.sep + '{}_path.txt'.format(seq)
    if not os.path.isfile(label_length_filepath):
        amount += 1
        print('Failed at Seq: {}'.format(seq))
print amount
