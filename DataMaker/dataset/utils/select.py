#! /usr/bin/env python
import os
from shutil import move, copy
import numpy as np

ose_folder = 'plans/ose'
gau_folder = 'plans/none'
label_folder = 'labels'
os.makedirs(label_folder) if not os.path.isdir(label_folder) else None
fileList = os.listdir('inputs')
data_size, path_too_long, path_too_complicate_or_simple = 0, [], []
ose_number, gau_number = 0, 0
for name in fileList:
    seq = name.split('_')[0]
    ose_length_filepath = ose_folder + os.sep + '{}_length.txt'.format(seq)
    gau_length_filepath = gau_folder + os.sep + '{}_length.txt'.format(seq)
    ose_length = np.loadtxt(ose_length_filepath) if os.path.isfile(ose_length_filepath) else [np.inf, np.inf]
    gau_length = np.loadtxt(gau_length_filepath) if os.path.isfile(gau_length_filepath) else [np.inf, np.inf]
    print seq, ose_length[0], gau_length[0]
    if ose_length[0] == np.inf and gau_length[0] == np.inf:
        print ('No Path for {}'.format(seq))
        continue
    if ose_length[0] <= gau_length[0]:
        source_filepath = ose_folder
        ose_number += 1
    else:
        source_filepath = gau_folder
        gau_number += 1
    path_filepath = source_filepath + os.sep + '{}_path.txt'.format(seq)
    path_output_filepath = label_folder + os.sep + '{}_path.txt'.format(seq)
    length_filepath = source_filepath + os.sep + '{}_length.txt'.format(seq)
    length_output_filepath = label_folder + os.sep + '{}_length.txt'.format(seq)

    path = np.loadtxt(path_filepath, delimiter=',')
    length = np.loadtxt(length_filepath, delimiter=',')
    if path.shape[0] - 2 > 5 or path.shape[0] - 2 == 0:
        path_too_complicate_or_simple.append(seq)
        print('Skipping {}, too C & S'.format(seq))
        continue
    if length[0] - length[1] > 20:
        path_too_long.append(seq)
        print('Skipping {}, too long'.format(seq))
        continue

    copy(path_filepath, path_output_filepath)
    copy(length_filepath, length_output_filepath)
    print(path_filepath + '-->>' + path_output_filepath)
    print(length_filepath + '-->>' + length_output_filepath)
    data_size += 1

print('Dataset Size {}'.format(data_size))
print('{} : {} = {} : {}'.format(ose_folder, gau_folder, ose_number, gau_number))
print('Too complicate {}'.format(len(path_too_complicate_or_simple)))
print('Too long {}'.format(len(path_too_long)))
print('Skipping amount: {}'.format(len(path_too_long) + len(path_too_complicate_or_simple)))
