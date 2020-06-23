#!/usr/bin/env python
import os
import numpy as np
import csv
import random

# configuration
data = []
train = []
valid = []
labels_folder = 'labels'
inputs_folder = 'inputs'
train_path = 'train.csv'
valid_path = 'valid.csv'
input_list = os.listdir(inputs_folder)
counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# set filepath of data
for i, name in enumerate(input_list):
    seq = name.split('_')[0]
    # cook x
    x_name = seq + '_encoded.png'
    x_path = inputs_folder + os.sep + x_name
    # cook y
    y_name = seq + '_path.txt'
    y_path = labels_folder + os.sep + y_name
    if os.path.isfile(y_path):
        data.append([int(seq), x_path, y_path])
        y_i = np.loadtxt(y_path, delimiter=',')
        counter[y_i.shape[0]-2] += 1
        if y_i.shape[0] == 2:
            print seq

# split dataset
data.sort()
for i, d in enumerate(data):
    if i % 4 == 0:
        valid.append(d[1:])
    else:
        train.append(d[1:])

# shuffle
random.shuffle(train)
random.shuffle(train)
random.shuffle(train)
random.shuffle(valid)
random.shuffle(valid)
random.shuffle(valid)

# save train set
with open(train_path, 'wb') as my_file:
    wr = csv.writer(my_file)
    for d in train:
        wr.writerow(d)
# save validation set
with open(valid_path, 'wb') as my_file:
    wr = csv.writer(my_file)
    for d in valid:
        wr.writerow(d)
print ('Dataset TOTAL SIZE: ' + str(len(os.listdir(labels_folder))))
print ("TrainSet SIZE: {}, ValidSet SIZE: {}".format(len(train), len(valid)))
print (counter)
