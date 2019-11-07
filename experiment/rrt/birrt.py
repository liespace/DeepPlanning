#!/usr/bin/env python
import time
import logging
import cProfile
import numpy as np
from planner import Planner
import cv2
from dtype import Location, State, Rotation, Velocity, TreeNode


def read_task(filepath, seq=0):
    """read source and target, and transform to right-hand"""
    task = np.loadtxt('{}/{}_task.txt'.format(filepath, seq), delimiter=',')
    org = task[0]
    org[2] = 0
    aim = task[1]
    org = State(location=Location(vec=[org[0], -org[1], 0.]),
                rotation=Rotation(rpy=(0., 0., -np.radians(org[3]))),
                velocity=Velocity())
    aim = State(location=Location(vec=[aim[0], -aim[1], 0.]),
                rotation=Rotation(rpy=(0., 0., -np.radians(aim[3]))),
                velocity=Velocity())
    return org, aim


def read_grid(filepath, seq):
    """read occupancy grid map"""
    return cv2.imread(filename='{}/{}_gridmap.png'.format(filepath, seq), flags=-1)


def write_way(seq, planner, folder='well'):
    """write way"""
    way = []
    if folder == 'well':
        for node in planner.propagator.path[0].path:
            state = planner.gridmap.origin.transform(node.state)
            way.append([state.location.x, state.location.y, state.rotation.y,
                        node.c2go.reverse])
        way = np.array(way)
        way = np.round(way, 4)
    np.savetxt('./dataset/{}/{}_way.txt'.format(folder, seq), way, delimiter=',')


def main():
    root = './dataset/blue'
    due, to, span, scale = 0, 1, 10, 1
    times, sill, bar = 25, 5, 3
    t_0 = time.time()
    for s in range((due * span) * scale, (to * span + 5) * scale):
        logging.info('Processing Situation %d', s)
        grid = read_grid(filepath=root, seq=s)
        org, aim = read_task(filepath=root, seq=s)
        path = []
        past = time.time()
        for i in range(times):
            logging.debug('Times: %d', i)
            planner = Planner()
            planner.director.aim = aim
            planner.gridmap.refresh(data=grid, seq=s, origin=org)
            planner.propagator.propagate()
            if planner.propagator.path:
                logging.debug('Path Cost: %f, Size: %d',
                              planner.propagator.path[0].c2go.cost(),
                              len(planner.propagator.path[0].path))
                path.append(planner.propagator.path[0])
            else:
                logging.debug('This time failed')
            if i == times-1 or (i == bar-1 and len(path) == bar) or len(path) >= sill:
                if path:
                    planner.propagator.path = path
                    planner.propagator.sort_path()
                    logging.info('\tPaths are: %s, size: %d',
                                 [np.around(n.c2go.cost(), 3) for n in path], len(path))
                    write_way(seq=s, planner=planner)
                    # planner.propagator.plot(filepath='{}/{}_prop'.format(root, s))
                    if planner.propagator.path[0].c2go.cost() > 35:
                        if i != times-1:
                            continue
                        else:
                            logging.warn('\tPath too long???')
                            write_way(seq=s, planner=planner, folder='warn')
                    if len(path) < 3:
                        logging.warning('\tNot enough paths!!?')
                        write_way(seq=s, planner=planner, folder='info')
                else:
                    logging.warning('\tNo Path!!!')
                    write_way(seq=s, planner=planner, folder='fail')

                now = time.time()
                logging.info('\tRuntime is: %.3f s of times: %d', now - past, i + 1)
                break
    t_1 = time.time()
    logging.info('Total Runtime is: %.3f m', (t_1 - t_0)/60.)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)
    logging.info("Bi-RRT Motion Planner Node is Running...")
    main()
    # pr = cProfile.Profile()
    # pr.enable()
    # main()
    # pr.disable()
    # pr.print_stats(sort='cumtime')
