#!/usr/bin/env python
import os
import re
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import reeds_shepp
from mpl_toolkits.mplot3d import Axes3D


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


def collision_check(states, grid_map):
    cons = [transform(contour(), state) for state in states]
    cons = [np.floor(con / 0.1 + 600 / 2.).astype(int) for con in cons]
    mask = np.zeros_like(grid_map, dtype=np.uint8)
    [cv2.fillPoly(mask, [con], 255) for con in cons]
    result = np.bitwise_and(mask, grid_map)
    collision_free = np.all(result < 255)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Mix", np.bitwise_and(np.bitwise_not(mask), grid_map) + mask)
    # cv2.waitKey()
    return collision_free


def pi_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def extract_path(filepath, seq, rho=5.0, size=0.1):
    ps = np.loadtxt('{}/{}/{}_path.txt'.format(filepath, '../labels', seq), delimiter=',')
    path = []
    for p in zip(ps[:-1], ps[1:]):
        path.extend(reeds_shepp.path_sample(p[0], p[1], rho, size))
    path = [p[:3] for p in path]
    print path
    return path


def state2config(state, x_min, y_min, t_min, xy_res, t_res):
    return (np.floor((state[0] - x_min) / xy_res),
            np.floor((state[1] - y_min) / xy_res),
            np.floor((pi_pi(state[2]) - t_min) / t_res))


def states2configs(states, x_min, y_min, t_min, xy_res, t_res):
    points = [state2config(s, x_min, y_min, t_min, xy_res, t_res) for s in states]
    return points


def set_plot():
    plt.ion()
    plt.figure()
    return plt.gca(projection='3d')


def main(filepath='./dataset/scenes', seq=0):
    # ax = set_plot()
    print('Processing {}'.format(seq))
    path = list(extract_path(filepath, seq))
    grid_map = read_grid(filepath, seq)
    nx, ny, nt = (300, 300, 300)
    x_min, x_max = -2., 28.
    y_min, y_max = -15., 15.
    t_min, t_max = -np.pi, np.pi
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    t = np.linspace(t_min, t_max, nt)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')
    space = np.zeros((nx, ny, nt))
    # plot start and goal configuration
    xy_res, t_res = (x_max - x_min) / nx, (t_max - t_min) / nt
    path = states2configs(path, x_min, y_min, t_min, xy_res, t_res)
    print path
    np.save('config_space/{}_path'.format(seq), path)
    print path[0], path[-1], xy_res, np.degrees(t_res)
    # raise Exception
    init_space = np.zeros_like(space)
    for p in path:
        px, py, pt = int(p[0]), int(p[1]), int(p[2])
        init_space[px, py, pt] = True
    np.save('config_space/{}_space_1'.format(seq), init_space)

    # ax.voxels(init_space, facecolors='g', edgecolor='k')
    for i in range(nx):
        print (i)
        for j in range(ny):
            for k in range(nt):
                state = (xx[i, j], yy[i, j], t[k])
                result = collision_check([state], grid_map)
                space[i, j, k] = not result
    # ax.voxels(space, facecolors=(0.8, 0.1, 0.1, 0.1), edgecolor=(0.8, 0.1, 0.1, 0.5))
    # plt.draw()
    # raw_input()
    np.save('config_space/{}_space'.format(seq), space)


if __name__ == '__main__':
    for no in [3520, 4960, 8320, 12320]:
        main(filepath='./dataset/scenes', seq=no)
