#!/usr/bin/env python
"""
Assemble all components of a rrt planner
"""
import csv
import os.path
import time
import rospkg
import rospy
from map import GridMap
from dtype import PlannerStatus
from propagation import BiPropagator
from sampling import GBSE2Sampler
from selection import WaySelector
from task import Director
from vehicle import Vehicle


class Planner(object):
    """
    Represent the planner
    DataFlow (U->D, L->R:
        OUTSIDE
            ||======||======||
        Vehicle     ||      ||
            |---GridMap     ||
            |       |----Directo
            |       |       |-------------------|
            |       |------------Sampler        |
            |-------|---------------|-----------|-Propagator
                                                    |-WaySelector
    """

    def __init__(self):
        self.vehicle = Vehicle()
        self.gridmap = GridMap(vehicle=self.vehicle)
        self.director = Director(gridmap=self.gridmap)
        self.sampler = GBSE2Sampler(gridmap=self.gridmap)
        self.propagator = BiPropagator(vehicle=self.vehicle,
                                       gridmap=self.gridmap,
                                       sampler=self.sampler,
                                       director=self.director)
        self.selector = WaySelector(propagator=self.propagator)
        self.seqs = {'VehicleStatus': None, 'GridMap': None}

    def run_once(self):
        """
        Run a cycle
        """
        if not self.ok():
            return PlannerStatus.STANDBY
        rospy.loginfo('Planning from ({}, {}, {})'.format(
            self.vehicle.status.point().x,
            self.vehicle.status.point().y,
            self.vehicle.status.point().z))
        if self.over():
            return PlannerStatus.ARRIVED

        t_0 = time.time()
        self.director.build_path()
        t_1 = time.time()
        self.propagator.propagate()
        t_2 = time.time()
        self.selector.search_way()
        t_3 = time.time()
        self.record_runtime(t_0, t_1, t_2, t_3)
        return PlannerStatus.RUNNING

    def ok(self):
        """
        check if material is ready
        """
        if self.director.route is None:
            rospy.logwarn('Route is Missing')
            return False

        if self.vehicle.status.seq is self.seqs['VehicleStatus']:
            rospy.logwarn('Vehicle Status is Missing')
            return False
        else:
            self.seqs['VehicleStatus'] = self.vehicle.status.seq

        if self.gridmap.seq is self.seqs['GridMap']:
            rospy.logwarn('Grid Map is Missing')
            return False
        else:
            self.seqs['GridMap'] = self.gridmap.seq

        rospy.loginfo('Materials are Ready')
        return True

    def over(self):
        """
        check if task if finished
        """
        if self.director.is_goal(location=self.vehicle.status.location):
            rospy.logwarn('GOT THE GOAL')
            return True
        return False

    @staticmethod
    def record_runtime(t_0, t_1, t_2, t_3):
        """
        record runtime of each component
        """
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('motion')
        root_path = os.path.abspath(
            os.path.join(pkg_path, os.pardir, os.pardir, os.pardir))
        with open('{}/{}/rrt_mplanner_runtime.csv'
                  .format(root_path, 'records'), 'ab') as f:
            writer = csv.writer(f)
            writer.writerow([int((t_1 - t_0) * 1e9),
                             int((t_2 - t_1) * 1e9),
                             int((t_3 - t_2) * 1e9)])
