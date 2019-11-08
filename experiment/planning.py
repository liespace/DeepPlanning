#!/usr/bin/env python
import time
import logging
import cProfile
import numpy as np
from rrt.planner import Planner
import cv2
from rrt.dtype import Location, State, Rotation, Velocity, TreeNode
import os
import glob


class PlanRunner(object):
    def __init__(self, file_type='valid', name='dw'):
        self.pred_filepath_root = 'pred/' + file_type
        self.true_filepath_root = 'dataset' + os.sep + 'well'
        self.cond_filepath_root = 'dataset' + os.sep + 'blue'
        self.plan_filepath_root = 'experiment' + os.sep + 'plan' + os.sep + name
        self.pred_folders = os.listdir(self.pred_filepath_root)

    def running(self, number, times=100):
        files, folder = self.find_files(number)
        ttf_ms, ttf_ss, cost_ms, cost_ss, sts, sf = [], [], [], [], [], 0
        for j, f in enumerate(files):
            print ('Processing Number {} and {}:'.format(j, f))
            # number of the example
            no = int(f.split('/')[-1].split('_')[0])
            grid = self.read_grid(no=no)
            org, aim = self.read_task(no=no)
            path, cost, ttf, st = [], [], [], 0
            for i in range(times):
                planner = Planner()
                planner.director.aim = aim
                planner.gridmap.refresh(data=grid, seq=no, origin=org)
                past = time.time()
                planner.propagator.propagate()
                now = time.time()
                if planner.propagator.path:
                    st += 1
                    ttf.append((now - past) * 1000)
                    path.append(planner.propagator.path[0])
                    cost.append(planner.propagator.path[0].c2go.cost())
                    # self.write_info(seq=no, planner=planner)
                    # planner.propagator.plot(filepath='')
            if st > 0:
                ttf_m, ttf_s = np.mean(ttf), np.sqrt(np.var(ttf))
                cost_m, cost_s = np.mean(cost), np.sqrt(np.var(ttf))
                self.write_info(no, ttf, cost, st)
                sf += 1
                sts.append(st)
                ttf_ms.append(ttf_m)
                ttf_ss.append(ttf_s)
                cost_ms.append(cost_m)
                cost_ss.append(cost_s)
                print (ttf_m, ttf_s, cost_m, cost_s, st)
            else:
                print (f + ' is FAILED')
            if j == 9:
                break
        self.write_result(ttf_ms, ttf_ss, cost_ms, cost_ss, sts, sf)
        print (np.mean(ttf_ms), np.mean(ttf_ss),
               np.mean(cost_ms), np.mean(cost_ss),
               np.mean(sts), sf)

    def write_result(self, ttf_ms, ttf_ss, cost_ms, cost_ss, sts, sf):
        ttf_ms_file = self.plan_filepath_root + os.sep + 'tff_ms.txt'
        ttf_ss_file = self.plan_filepath_root + os.sep + 'tff_ss.txt'
        cost_ms_file = self.plan_filepath_root + os.sep + 'cost_ms.txt'
        cost_ss_file = self.plan_filepath_root + os.sep + 'cost_ss.txt'
        sts_file = self.plan_filepath_root + os.sep + 'sts.txt'
        sf_file = self.plan_filepath_root + os.sep + 'sf.txt'
        np.savetxt(ttf_ms_file, np.array(ttf_ms), delimiter=',')
        np.savetxt(ttf_ss_file, np.array(ttf_ss), delimiter=',')
        np.savetxt(cost_ms_file, np.array(cost_ms), delimiter=',')
        np.savetxt(cost_ss_file, np.array(cost_ss), delimiter=',')
        np.savetxt(sts_file, np.array(sts), delimiter=',')
        np.savetxt(sf_file, np.array([sf]), delimiter=',')

    def write_info(self, seq, ttf, cost, st):
        ttf_file = self.plan_filepath_root + os.sep + str(seq) + '_tff.txt'
        cost_file = self.plan_filepath_root + os.sep + str(seq) + '_cost.txt'
        st_file = self.plan_filepath_root + os.sep + str(seq) + '_st.txt'
        np.savetxt(ttf_file, np.array(ttf), delimiter=',')
        np.savetxt(cost_file, np.array(cost), delimiter=',')
        np.savetxt(st_file, np.array([st]), delimiter=',')

    def write_way(self, seq, planner):
        """write way"""
        if not os.path.isdir(self.plan_filepath_root):
            os.makedirs(self.plan_filepath_root)
        path = []
        for node in planner.propagator.path[0].path:
            state = planner.gridmap.origin.transform(node.state)
            path.append(
                [state.location.x, state.location.y, state.rotation.y,
                 node.c2go.reverse])
        path = np.array(path)
        path = np.round(path, 4)
        filename = self.plan_filepath_root + os.sep + str(seq) + '_way.txt'
        np.savetxt(filename, path, delimiter=',')

    def read_task(self, no):
        """read source and target, and transform to right-hand"""
        task_file = self.cond_filepath_root + os.sep + str(no) + '_task.txt'
        task = np.loadtxt(task_file, delimiter=',')
        org, aim = task[0], task[1]
        org = State(location=Location(vec=[org[0], -org[1], 0.]),
                    rotation=Rotation(rpy=(0., 0., -np.radians(org[3]))),
                    velocity=Velocity())
        aim = State(location=Location(vec=[aim[0], -aim[1], 0.]),
                    rotation=Rotation(rpy=(0., 0., -np.radians(aim[3]))),
                    velocity=Velocity())
        return org, aim

    def read_grid(self, no):
        """read occupancy grid map"""
        grid_file = self.cond_filepath_root + os.sep + str(no) + '_gridmap.png'
        return cv2.imread(filename=grid_file, flags=-1)

    def find_files(self, number):
        for folder in self.pred_folders[number: number+1]:
            pred_filepath = self.pred_filepath_root + os.sep + folder
            print("Processing " + pred_filepath)
            files = glob.glob(pred_filepath + os.sep + '*.txt')
            return files, folder


if __name__ == '__main__':
    runner = PlanRunner(file_type='valid', name='rrt')
    runner.running(3)
    # pr = cProfile.Profile()
    # pr.enable()
    # main()
    # pr.disable()
    # pr.print_stats(sort='cumtime')
