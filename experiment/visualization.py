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


def plot_tri(
        q, radius=0.8, color='r', fill=True, width=2, zorder=0, poster=plt):
    v1 = get_point(q[:-1], radius=radius, orin=q[2])
    v2 = get_point(q[:-1], radius=radius, orin=q[2] + 120. / 180. * np.pi)
    v3 = get_point(q[:-1], radius=radius, orin=q[2] - 120. / 180. * np.pi)
    if fill:
        tri = np.array([v1, v2, v3])
        poster.fill(
            tri[:, 0], tri[:, 1], color=color, zorder=zorder)
    else:
        tri = np.array([v1, v2, v3, v1])
        poster.plot(
            tri[:, 0], tri[:, 1], color=color, linewidth=width, zorder=zorder)
    poster.scatter(q[0], q[1], color=color, s=50, zorder=zorder)


def plot_keys(curve, width=4.8, height=2., zorder=10, alpha=0.5,
              color='g', linewidth=1, poster=plt, fill=False):
    a = float(alpha)
    for p in curve:
        q = p[0:3]
        patch = poster.Rectangle(
            xy=(q[0] - width / 2., q[1] - height / 2.),
            color=color, fill=fill, linewidth=linewidth, zorder=zorder,
            width=width, height=height, angle=np.degrees(q[2]), alpha=a)
        alpha = np.arctan2(height, width) + q[2]
        norm = np.sqrt((width / 2.) ** 2 + (height / 2.) ** 2)
        patch.set_xy(
            xy=(q[0] - norm * np.cos(alpha), q[1] - norm * np.sin(alpha)))
        poster.gca().add_patch(patch)


def plot_curve(curve, zorder=200, linewidth=3,
               color='r', style='-', poster=plt):
    xs = [q[0] for q in curve]
    ys = [q[1] for q in curve]
    poster.plot(
        xs, ys, color=color, linestyle=style,
        linewidth=linewidth, zorder=zorder)
    poster.axis('equal')


def plot_vehicles(
        cps, width=4.8, height=2., zorder=0, alpha=0.2,
        fill=False, color='g', linewidth=2, poster=plt):
    a = float(alpha)
    for p in cps[1:-1]:
        q = p[0:3]
        # patch = poster.Rectangle(
        #     xy=(q[0], q[1]), fill=fill, color=color, linewidth=linewidth,
        #     width=width, height=height, angle=np.degrees(q[2]), zorder=zorder,
        #     alpha=a)
        # alpha = np.arctan2(height, width) + q[2]
        # norm = np.sqrt((width/2.)**2 + (height/2.)**2)
        # patch.set_xy(xy=(q[0]-norm*np.cos(alpha), q[1]-norm*np.sin(alpha)))
        # poster.gca().add_patch(patch)

        plot_tri(
            q, radius=1.0, color=color, zorder=zorder,
            fill=False, width=linewidth, poster=poster)


def plot_path(
        cps, curve=True, keys=True, car=True,
        step_size=0.5, rho=5., width=4.8, height=2.,
        car_color='b', car_fill=False, car_width=2, car_zorder=0, car_alpha=0.2,
        key_color='r', key_width=2, key_zorder=10, key_fill=False, key_alpha=0.2,
        curve_color='g', curve_style='-', curve_width=2, curve_zorder=100,
        poster=plt):
    c = build_curve(cps=cps, step_size=step_size, rho=rho)
    if curve:
        plot_curve(
            curve=c, zorder=curve_zorder, linewidth=curve_width,
            color=curve_color, style=curve_style, poster=poster)
    if keys:
        plot_keys(
            curve=c, width=width, height=height, zorder=key_zorder,
            alpha=key_alpha, color=key_color, linewidth=key_width,
            poster=poster, fill=key_fill)
    if car:
        plot_vehicles(
            cps=cps, width=width, height=height, zorder=car_zorder, alpha=car_alpha,
            fill=car_fill, color=car_color, linewidth=car_width, poster=poster)


def build_curve(cps, step_size=0.5, rho=5.):
    curve = []
    for i in range(cps.shape[0] - 1):
        q0 = cps[i][0:3]
        q1 = cps[i + 1][0:3]
        curve.extend(reeds_shepp.path_sample(q0, q1, rho, step_size))
    curve.append([cps[-1][0], cps[-1][1], cps[-1][2]])
    return curve


def build_dubins(cps, step_size=0.5, rho=5.):
    curve = []
    for i in range(cps.shape[0] - 1):
        q0, q1 = cps[i][0:3], cps[i+1][0:3]
        mask = cps[i+1][3]
        if mask == 0:
            p = dubins.shortest_path(q0, q1, rho)
        else:
            p = dubins.shortest_path(q1, q0, rho)
        configurations, _ = p.sample_many(step_size)
        curve.extend(configurations)
    return curve


def plot_true(
        cps, curve=True, keys=True, car=True,
        step_size=0.5, rho=5., width=4.8, height=2.,
        car_color='b', car_fill=False, car_width=2, car_zorder=0, car_alpha=0.2,
        key_color='r', key_width=2, key_zorder=10, key_fill=False, key_alpha=0.2,
        curve_color='g', curve_style='-', curve_width=2, curve_zorder=100,
        poster=plt):
    for i in range(cps.shape[0] - 1):
        q0, q1 = cps[i][0:3], cps[i+1][0:3]
        mask = cps[i+1][3]
        if mask == 0:
            p = dubins.shortest_path(q0, q1, rho)
            c, _ = p.sample_many(step_size)
            c.append(q1)
        else:
            p = dubins.shortest_path(q1, q0, rho)
            c, _ = p.sample_many(step_size)
            c.append(q0)
        if curve:
            plot_curve(
                curve=c, zorder=curve_zorder, linewidth=curve_width,
                color=curve_color, style=curve_style, poster=poster)
        if keys:
            plot_keys(
                curve=c, width=width, height=height, zorder=key_zorder,
                alpha=key_alpha, color=key_color, linewidth=key_width,
                poster=poster, fill=key_fill)
        if car:
            plot_vehicles(
                cps=cps, width=width, height=height, zorder=car_zorder, alpha=car_alpha,
                fill=car_fill, color=car_color, linewidth=car_width, poster=poster)


def plot_grid(grid, res=0.1, wic=600, hic=600,
              dot_size=30., marker='s', zorder=0, poster=plt):
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
    poster.scatter(x='x', y='y', c='c',
                   data=data, s=dot_size, marker=marker, zorder=zorder)


def plot_cond(
        cps, colors=('b', 'g'), width=4.7, height=2.0,
        zorder=0, linewidth=2, fill=True, alpha=0.2, poster=plt):
    a = float(alpha)
    states = [cps[0], cps[-1]]
    for i, p in enumerate(states):
        q = p[0:3]
        patch = poster.Rectangle(
            xy=(q[0] - width / 2., q[1] - height / 2.),
            color=colors[i], fill=fill, linewidth=linewidth, zorder=zorder,
            width=width, height=height, angle=np.degrees(q[2]), alpha=a)
        alpha = np.arctan2(height, width) + q[2]
        norm = np.sqrt((width / 2.) ** 2 + (height / 2.) ** 2)
        patch.set_xy(
            xy=(q[0] - norm * np.cos(alpha), q[1] - norm * np.sin(alpha)))
        poster.gca().add_patch(patch)

        plot_tri(
            q, radius=1.0, color=colors[i], zorder=zorder,
            fill=False, width=linewidth, poster=poster)


def test(no=0, well_path='./dataset/well'):
    filename = well_path + os.sep + str(no) + '_way.txt'
    cps = np.loadtxt(filename, delimiter=',')
    plot_path(cps, step_size=0.5, rho=5., width=4.8, height=2.,
              car_color='b', car_fill=False, car_width=4, car_zorder=0,
              key_color='r', key_width=1, key_zorder=10,
              curve_color='r', curve_style='-', curve_width=4, curve_zorder=200,
              poster=plt)
    plot_cond(
        cps, colors=('b', 'g'), width=4.7, height=2.0,
        zorder=5, linewidth=2, fill=True, alpha=0.4, poster=plt)
    plot_true(cps, step_size=0.5, rho=5., width=4.8, height=2.,
              car_color='b', car_fill=False, car_width=4, car_zorder=0,
              key_color='g', key_width=1, key_zorder=10,
              curve_color='r', curve_style='--', curve_width=4, curve_zorder=200,
              poster=plt)


if __name__ == '__main__':
    test(no=8280)
    plt.show()
