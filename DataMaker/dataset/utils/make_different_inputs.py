#!/usr/bin/env python
import os
import re
import time
import cv2
import numpy as np


def gcs2lcs(state, origin):
    xo, yo, ao = origin[0], origin[1], origin[2]
    x = (state[0] - xo) * np.cos(ao) + (state[1] - yo) * np.sin(ao)
    y = -(state[0] - xo) * np.sin(ao) + (state[1] - yo) * np.cos(ao)
    a = state[2] - ao
    return np.array((x, y, a))


def polygon(wheelbase=0., width=2.116 + 0.2, length=4.925 + 0.2):  # 2.96, 2.2, 4.925
    return np.array([
        [-(length / 2. - wheelbase / 2.), width / 2. - 1.0], [-(length / 2. - wheelbase / 2. - 0.4), width / 2.],
        [length / 2. + wheelbase / 2. - 0.6, width / 2.], [length / 2. + wheelbase / 2., width / 2. - 0.8],
        [length / 2. + wheelbase / 2., -(width / 2. - 0.8)], [length / 2. + wheelbase / 2. - 0.6, -width / 2.],
        [-(length / 2. - wheelbase / 2. - 0.4), -width / 2.], [-(length / 2. - wheelbase / 2.), -(width / 2. - 1.0)]])


def rectangle(wheelbase=0., width=2.116 + 0.2, length=4.925 + 0.2):  # 2.96, 2.2, 4.925
    return np.array([
        [-(length / 2. - wheelbase / 2.), width / 2.], [length / 2. + wheelbase / 2., width / 2.],
        [length / 2. + wheelbase / 2., -width / 2.], [-(length / 2. - wheelbase / 2.), -width / 2.]])


def transform(poly, pto):
    pts = poly.transpose()
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return (np.dot(rot, pts) + xyo).transpose()


def read_task(filepath, seq=0):
    """
    read source(start) and target(goal), and transform to right-hand and local coordinate system centered in source
    LCS: local coordinate system, or said vehicle-frame.
    GCS: global coordinate system
    """
    # read task and transform coordinate system to right-hand
    task = np.loadtxt('{}/{}_task.txt'.format(filepath, seq), delimiter=',')
    org, aim = task[0], task[1]
    # coordinate of the center of mass on source(start) state, in GCS
    source = (org[0], -org[1], -np.radians(org[3]))
    # coordinate of center of mass on target(goal) state, in GCS
    target = (aim[0], -aim[1], -np.radians(aim[3]))
    return source, target


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return cv2.imread('{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


def read_input(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    img = cv2.imread('{}/{}_encoded.png'.format(filepath, seq), flags=-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_seqs(filepath):
    file_list = os.listdir(filepath)
    return [re.sub('\\D', '', f.strip().split(',')[0]) for f in file_list]


def draw_state(layer, state, value, width, length, draw_type, grid_res=0.1, grid_size=600):
    if draw_type == 'circle':
        radius = int(width / 0.2)
        center = np.floor(state[:2] / grid_res + grid_size / 2.).astype(int)
        layer = cv2.circle(layer, tuple(center), radius, value, -1)
    elif draw_type == 'rectangle':
        con = np.floor(transform(rectangle(width=width, length=length), state) / grid_res + grid_size / 2.).astype(int)
        layer = cv2.fillPoly(layer, [con], value)
    else:
        con = np.floor(transform(polygon(width=width, length=length), state) / 0.1 + 600 / 2.).astype(int)
        layer = cv2.fillPoly(layer, [con], value)
    return layer


def main(dataset_folder='.', draw_type='rectangle', include_start=False, include_unknown=False, tag='input2',
         width=2.116+0.2, length=4.925+0.2):
    output_filepath = dataset_folder + os.sep + tag
    scenes_filepath = dataset_folder + os.sep + 'scenes'
    inputs_filepath = dataset_folder + os.sep + 'inputs'
    seqs = read_seqs(inputs_filepath)
    # width, length = 2.116 + 0.2, 4.925 + 0.2
    for i, seq in enumerate(seqs):
        print('Processing {} ({} of {})'.format(seq, i, len(seqs)))
        grid_map = read_grid(scenes_filepath, seq)
        source, target = read_task(scenes_filepath, seq)
        start = gcs2lcs(source, source)
        goal = gcs2lcs(target, source)

        loc_layer = np.zeros_like(grid_map, dtype=np.uint8)
        loc_layer = draw_state(loc_layer, np.array(goal), 255, width, length, draw_type)
        if include_start:
            loc_layer = draw_state(loc_layer, np.array(start), 127, width, length, draw_type)

        ori_layer = np.zeros_like(grid_map, dtype=np.uint8)
        angle = (goal[2] + np.pi) % (2 * np.pi) - np.pi
        angle = (angle + np.pi) / (2. * np.pi) * 255.
        if angle > 360:
            print seq, angle
        ori_layer = draw_state(ori_layer, np.array(goal), int(angle), width, length, draw_type)
        if include_start:
            ori_layer = draw_state(ori_layer, np.array(start), 0, width, length, draw_type)

        obs_layer = np.bitwise_and(grid_map, np.bitwise_not(loc_layer))
        if not include_unknown:
            obs_layer = obs_layer / 255 * 255

        rgb_array = np.zeros((grid_map.shape[0], grid_map.shape[1], 3), dtype=np.uint8)
        rgb_array[:, :, 0] = obs_layer[:]
        rgb_array[:, :, 1] = loc_layer[:]
        rgb_array[:, :, 2] = ori_layer[:]
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        ori_input = read_input(inputs_filepath, seq)
        os.makedirs(output_filepath) if not os.path.isdir(output_filepath) else None
        cv2.imwrite('{}{}{}_encoded.png'.format(output_filepath, os.sep, seq), bgr_array)
        # cv2.imshow("Mask", bgr_array)
        # cv2.waitKey()


if __name__ == '__main__':
    main(dataset_folder='.',
         draw_type='rectangle',  # circle, polygon
         include_start=True,
         include_unknown=False,
         tag='inputs',
         width=(2.116+0.2)/1.,  # default: (2.116+0.2)
         length=(4.925+0.2)/1.)  # default: (4.925+0.2)
# 0: r-g-o-u-s
# 1: rs-g-o-u-s, width /=2, length /= 2
# 2: rb-g-o-u-s, width *=2, length /= 2
# 3: cb-g-o-u-s, width = 1.944 * 2
# 4: cs-g-o-u-s
# 5: p-g-o-u-s
# 6: r-g-o-u
# 7: r-g-o-s
