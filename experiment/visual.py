#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import reeds_shepp
import dubins


def get_point(center, radius, orin):
    x = center[0] + radius * np.cos(orin)
    y = center[1] + radius * np.sin(orin)
    return x, y


def plot_path(path, line_color='g', point_color='r'):
    xs = [q[0] for q in path]
    ys = [q[1] for q in path]
    plt.plot(xs, ys, line_color+'-')
    plt.plot(xs, ys, point_color+'.')
    plt.axis('equal')


def plot_car(q, size, color='g'):
    a = get_point(q[:-1], size, q[2])
    b = get_point(q[:-1], size / 2, q[2] + 150. / 180. * np.pi)
    c = get_point(q[:-1], size / 2, q[2] - 150. / 180. * np.pi)
    tri = np.array([a, b, c, a])
    plt.plot(tri[:, 0], tri[:, 1], color+'-')


def plot_keys(way, size, color='g'):
    for p in way:
        q = p[0:3]
        circle = plt.Circle(xy=(q[0], q[1]), radius=size/15, color=color)
        plt.gca().add_patch(circle)
        plot_car(q, size, color=color)


def plot_way(way, step_size, rho=5., car_size=1.,
             car_color='b', line_color='g', point_color='r'):
    num = way.shape[0]
    path = []
    for i in range(num-1):
        q0 = way[i][0:3]
        q1 = way[i+1][0:3]
        path.extend(reeds_shepp.path_sample(q0, q1, rho, step_size))
    path.append([way[-1][0], way[-1][1]])
    plot_path(path, line_color=line_color, point_color=point_color)
    plot_keys(way=way, size=car_size, color=car_color)


def plot_dubins(way, step_size, rho=5., car_size=1., mask=(0, 1),
                car_color='b', line_color='g', point_color='r'):
    num = way.shape[0]
    path = []
    for i in range(num-1):
        q0 = way[i][0:3]
        q1 = way[i+1][0:3]
        if mask[i] == 0:
            p = dubins.shortest_path(q0, q1, rho)
        else:
            p = dubins.shortest_path(q1, q0, rho)
        configurations, _ = p.sample_many(step_size)
        path.extend(configurations)
    plot_path(path, line_color=line_color, point_color=point_color)
    plot_keys(way=way, size=car_size, color=car_color)


def plot_grid(grid, res=0.1, wic=600, hic=600):
    """plot grid map"""
    row, col = grid.shape[0], grid.shape[1]
    u = np.array(range(row)).repeat(col)
    v = np.array(range(col) * row)
    uv = np.array([u, v, np.ones_like(u)])
    xy2uv = np.array([[0., 1. / res, hic / 2.],
                     [1. / res, 0., wic / 2.],
                     [0., 0., 1.]])
    xy = np.dot(np.linalg.inv(xy2uv), uv)
    data = {'x': xy[0, :], 'y': xy[1, :],
            'c': np.array(grid).flatten() - 1}
    plt.scatter(x='x', y='y', c='c', data=data, s=30., marker="s")


def test(no=0, well_path='./well'):
    filename = well_path + os.sep + str(no) + '_way.txt'
    way = np.loadtxt(filename, delimiter=',')
    plot_way(way=way, step_size=0.5, rho=5.)


if __name__ == '__main__':
    test(no=8200)
