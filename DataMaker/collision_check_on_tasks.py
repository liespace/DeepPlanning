#!/usr/bin/env python
import os
import re
import time
import cv2
import numpy as np


def center2rear(node, wheelbase=2.96):
    """calculate the coordinate of rear track center according to mass center"""
    theta, r = node[2] + np.pi, wheelbase / 2.
    node[0] += r * np.cos(theta)
    node[1] += r * np.sin(theta)
    return node


def gcs2lcs(state, origin):
    xo, yo, ao = origin[0], origin[1], origin[2]
    x = (state[0] - xo) * np.cos(ao) + (state[1] - yo) * np.sin(ao)
    y = -(state[0] - xo) * np.sin(ao) + (state[1] - yo) * np.cos(ao)
    a = state[2] - ao
    return np.array((x, y, a))


def contour(wheelbase=2.850, width=2.116 + 0.2, length=4.925 + 0.2):  # 2.96, 2.2, 4.925
    return np.array([
        [-(length/2. - wheelbase / 2.), width/2. - 1.0], [-(length/2. - wheelbase / 2. - 0.4), width/2.],
        [length/2. + wheelbase / 2. - 0.6, width/2.], [length/2. + wheelbase / 2., width/2. - 0.8],
        [length/2. + wheelbase / 2., -(width/2. - 0.8)], [length/2. + wheelbase / 2. - 0.6, -width/2.],
        [-(length/2. - wheelbase / 2. - 0.4), -width/2.], [-(length/2. - wheelbase / 2.), -(width/2. - 1.0)]])


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
    source =(org[0], -org[1], -np.radians(org[3]))
    # coordinate of center of mass on target(goal) state, in GCS
    target = (aim[0], -aim[1], -np.radians(aim[3]))
    return source, target


def read_grid(filepath, seq):
    # type: (str, int) -> np.ndarray
    """read occupancy grid map"""
    return cv2.imread('{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


def main():
    file_list = os.listdir('./dataset/inputs/')
    seqs = [re.sub('\\D', '', f.strip().split(',')[0]) for f in file_list]
    failed_number = 0
    for seq in seqs:
        # print('Processing {}'.format(seq))
        filepath = './dataset/scenes'
        source, target = read_task(filepath, seq)
        grid_map = read_grid(filepath, seq)
        start = center2rear(gcs2lcs(source, source))
        goal = center2rear(gcs2lcs(target, source))

        # states = reeds_shepp.path_sample(start, goal, 5.0, 0.3)
        states = [start, goal]
        cons = [transform(contour(), state) for state in states]
        cons = [np.floor(con / 0.1 + 600 / 2.).astype(int) for con in cons]

        mask = np.zeros_like(grid_map, dtype=np.uint8)
        past = time.time()
        [cv2.fillPoly(mask, [con], 255) for con in cons]
        now = time.time()
        # print ((now - past) * 1000)
        result = np.bitwise_and(mask, grid_map)
        collision_free = np.all(result < 255)
        if not collision_free:
            # if not seq[:2] == '12' and not seq[:2] == '93':
            print('Failed at {}'.format(seq))
            failed_number += 1
        print (np.all(result < 255))
        cv2.imshow("Mask", mask)
        cv2.imshow("Mix", np.bitwise_and(np.bitwise_not(mask), grid_map) + mask)
        cv2.waitKey()
    print(failed_number)


if __name__ == '__main__':
    main()
