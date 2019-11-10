#!/usr/bin/env python
import time
import logging
import cProfile
import numpy as np
from rrt.planner import Planner, DWAPlanner
import cv2
from rrt.dtype import Location, State, Rotation, Velocity, TreeNode
import os
import glob
import sys
from planning import PlanRunner
import visual


class DWPlanRunner(PlanRunner):
    def __init__(self, file_type='valid', name='dwa-rrt', recall_bar=0.7):
        super(DWPlanRunner, self).__init__(file_type=file_type, name=name)
        self.recall_bar = recall_bar

    def running(self, number, index=0, processing=4):
        files, folder = self.find_files(number)
        step = int(np.ceil(len(files) / float(processing)))
        self.plan_filepath_root += (os.sep + folder + '_' + str(self.recall_bar))
        self.run_one(files[index * step:(index + 1) * step], 100)

    def run_one(self, files, times=100):
        ttf_ms, ttf_ss, cost_ms, cost_ss, sts, sf = [], [], [], [], [], 0
        cost2_ms, cost2_ss = [], []
        for j, f in enumerate(files):
            print ('Processing Number {}/{} and {}:'.format(j, len(files), f))
            # number of the example
            no = int(f.split('/')[-1].split('_')[0])
            grid = self.read_grid(no=no)
            org, aim = self.read_task(no=no)
            reference = self.read_referential_path(f, no)
            path, cost, ttf, st, cost2 = [], [], [], 0, []
            for i in range(times):
                planner = DWAPlanner()
                planner.director.aim = aim
                planner.director.goal = aim.location
                planner.director.path = reference
                planner.gridmap.refresh(data=grid, seq=no, origin=org)
                past = time.time()
                planner.propagator.propagate()
                now = time.time()
                planner.selector.search_way()
                if planner.propagator.anode:
                    st += 1
                    ttf.append((now - past) * 1000)
                    path.append(planner.propagator.anode)
                    cost.append(planner.propagator.anode.c2go.cost())

                    planner.propagator.extend = -100
                    planner.propagator.is_first = False
                    planner.propagator.propagate()
                    planner.selector.search_way()
                    cost2.append(planner.propagator.anode.c2go.cost())
                    # plot
                    # planner.propagator.path.append(planner.propagator.anode)
                    # planner.propagator.plot(filepath='')
            if st > 0:
                ttf_m, ttf_s = np.mean(ttf), np.sqrt(np.var(ttf))
                cost_m, cost_s = np.mean(cost), np.sqrt(np.var(cost))
                cost2_m, cost2_s = np.mean(cost2), np.sqrt(np.var(cost2))
                self.write_info(no, ttf, cost, st, cost2)
                sf += 1
                sts.append(st)
                ttf_ms.append(ttf_m)
                ttf_ss.append(ttf_s)
                cost_ms.append(cost_m)
                cost_ss.append(cost_s)
                cost2_ms.append(cost2_m)
                cost2_ss.append(cost2_s)
                print (ttf_m, ttf_s, cost_m, cost_s, st, cost2_m, cost2_s)
            else:
                print (f + ' is FAILED')
            if j == 10:
                break
        self.write_result(
            ttf_ms, ttf_ss, cost_ms, cost_ss, sts, sf, cost2_ms, cost2_ss)
        print (np.mean(ttf_ms), np.mean(ttf_ss),
               np.mean(cost_ms), np.mean(cost_ss),
               np.mean(cost2_ms), np.mean(cost2_ss),
               np.mean(sts), sf)

    def read_task(self, no):
        """read source and target, and transform to right-hand"""
        true_file = self.true_filepath_root + os.sep + str(no) + '_way.txt'
        true = np.loadtxt(true_file, delimiter=',')
        org, aim = true[0], true[-1]
        org = State(location=Location(vec=[org[0], org[1], 0.]),
                    rotation=Rotation(rpy=(0., 0., org[2])),
                    velocity=Velocity())
        aim = State(location=Location(vec=[aim[0], aim[1], 0.]),
                    rotation=Rotation(rpy=(0., 0., aim[2])),
                    velocity=Velocity())
        return org, aim

    def read_referential_path(self, f, no):
        true_file = self.true_filepath_root + os.sep + str(no) + '_way.txt'
        true = np.loadtxt(true_file, delimiter=',')
        true = true[:, :-1]
        pred = np.loadtxt(f, delimiter=',')
        p_prt, path = [], []
        for row in pred:
            if row[-1] > self.recall_bar:
                p_prt.append(row[:-1])
        p_prt.append(true[-1])
        p_prt.insert(0, true[0])
        # visual.plot_way(np.array(p_prt), step_size=0.5,
        #                 car_color='b', line_color='b', point_color='b')
        # visual.plot_way(np.array(true), step_size=0.5,
        #                 car_color='g', line_color='g', point_color='g')
        for p in p_prt[1:]:
            state = State(location=Location(vec=[p[0], p[1], 0.]),
                          rotation=Rotation(rpy=(0., 0., p[2])),
                          velocity=Velocity())
            path.append(state)
        return path


if __name__ == '__main__':
    runner = DWPlanRunner(file_type='valid', name='dwa-rrt', recall_bar=0.8)
    needed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    thread = 4 if len(sys.argv) > 1 else 1
    runner.running(number=2, index=needed, processing=thread)
    # pr = cProfile.Profile()
    # pr.enable()
    # main()
    # pr.disable()
    # pr.print_stats(sort='cumtime')

    # First 10 Examples and Fitted Gaussian
    # 2 vgg19_comp_free200_check300 - 0.7 ** / 0.8 *
    # 3 vgg19_comp_free200_check400 - 0.7 *** / 0.8 ****

    # First 10 Examples and unFitted Gaussian
    # 2 vgg19_comp_free200_check300 - 0.7 * / 0.8 *
    # 3 vgg19_comp_free200_check400 - 0.7 * / 0.8 *
