#!/usr/bin/env python
import logging
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from SORRT.planner import RRTStar, BiRRTStar
from SORRT.debugger import Debugger


def center2rear(node, wheelbase=2.850):
    """calculate the coordinate of rear track center according to mass center"""
    if not isinstance(node, RRTStar.StateNode):
        theta, r = node[2] + np.pi, wheelbase / 2.
        node[0] += r * np.cos(theta)
        node[1] += r * np.sin(theta)
        return node
    theta, r = node.state[2] + np.pi, wheelbase / 2.
    node.state[0] += r * np.cos(theta)
    node.state[1] += r * np.sin(theta)
    return node


def contour(wheelbase=2.850, width=2.116 + 0.4, length=4.925 + 0.4):  # 2.96, 2.2, 5.0
    return np.array([
        [-(length/2. - wheelbase / 2.), width/2. - 1.0], [-(length/2. - wheelbase / 2. - 0.4), width/2.],
        [length/2. + wheelbase / 2. - 0.6, width/2.], [length/2. + wheelbase / 2., width/2. - 0.8],
        [length/2. + wheelbase / 2., -(width/2. - 0.8)], [length/2. + wheelbase / 2. - 0.6, -width/2.],
        [-(length/2. - wheelbase / 2. - 0.4), -width/2.], [-(length/2. - wheelbase / 2.), -(width/2. - 1.0)]])


def read_yips(filepath, seq, discrimination=0.7):
    yips = np.loadtxt('{}/{}_pred.txt'.format(filepath, seq), delimiter=',')
    yips = filter(lambda x: x[-1] > discrimination, yips)
    yips = map(center2rear, yips)
    yips = [((yip[0], yip[1], yip[2]), ((0.621, 2.146), (0.015, 1.951 * 1.0), (0.005, 0.401 * 1.0))) for yip in yips]
    return yips


def set_plot(switch=True):
    if switch:
        plt.ion()
        plt.figure()
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().set_aspect('equal')
        plt.gca().set_facecolor((0.2, 0.2, 0.2))
        plt.gca().set_xlim((-30, 30))
        plt.gca().set_ylim((-30, 30))
        plt.draw()


def transform(pts, pto):
    xyo = np.array([[pto[0]], [pto[1]]])
    rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
    return np.dot(rot, pts) + xyo


def carla_transform_to_node(carla_trans):
    return RRTStar.StateNode(
        state=(carla_trans.location.x, -carla_trans.location.y, -np.radians(carla_trans.rotation.yaw)))


def yips_path_to_heuristic(yips_path):
    yips = [((s[0], s[1], s[2]), ((0.047, 2.442), (0.074, 1.723), (-0.011, 0.512))) for s in yips_path]
    return yips


def optimize_path(source, target, grid_map, yips_path, debug=False):
    rrt_star = BiRRTStar().set_vehicle(contour(), 0.3, 0.20)
    heuristic = yips_path_to_heuristic(yips_path)
    ori = carla_transform_to_node(source)
    start = center2rear(carla_transform_to_node(source)).gcs2lcs(ori.state)
    goal = center2rear(carla_transform_to_node(target)).gcs2lcs(ori.state)
    grid_ori, grid_res = deepcopy(ori).gcs2lcs(ori.state), 0.1
    if debug:
        set_plot(debug)
        Debugger.plot_grid(grid_map, grid_res)
        Debugger().plot_nodes([start, goal])
        plt.gca().add_patch(Polygon(
            transform(contour().transpose(), start.state).transpose(), True, color='b', fill=False, lw=2.0))
        plt.gca().add_patch(Polygon(
            transform(contour().transpose(), goal.state).transpose(), True, color='g', fill=False, lw=2.0))
        if heuristic:
            Debugger.plot_heuristic(heuristic)
        plt.draw()
    rrt_star.debug = debug
    rrt_star.preset(start, goal, grid_map, grid_res, grid_ori, 255, heuristic).planning(500)
    while not rrt_star.x_best.fu < np.inf:
        logging.warning('Warning, Hard Problem')
        rrt_star.preset(start, goal, grid_map, grid_res, grid_ori, 255, heuristic).planning(500*4)
    tj = rrt_star.trajectory(a_cc=3, v_max=10, res=0.1)
    plt.plot([t.state[0] for t in tj], [t.state[1] for t in tj])
    plt.scatter([t.state[0] for t in tj], [t.state[1] for t in tj], c=[t.k for t in tj], s=50)
    # LCS to GCS
    tj = [t.lcs2gcs(ori.state) for t in tj]
    motion = [(t.state[0], t.state[1], t.state[2], t.k, t.v) for t in tj]
    Debugger.breaker('Plotting', switch=debug)
    return motion, [tuple(p.state) for p in rrt_star.path()]
