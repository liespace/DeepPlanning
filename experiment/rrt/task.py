#!/usr/bin/env python
"""
Module to handle planning task generation
"""
from collections import namedtuple
from typing import List  # pylint: disable=unused-import
from geomdl import BSpline
import numpy as np
import rospy
from tabulate import tabulate
from map import GridMap
from dtype import State, Location, Rotation, Velocity


PathConfig = namedtuple('PathConfig', ['scope', 'wmc', 'precision', 'order'])


class Director(object):  # pylint: disable=too-many-instance-attributes
    """
    Class handle route and path generation.
        > config: PathConfig('scope', 'wmc', 'precision', 'order')
            > scope: only the points in the circle (center is vehicle, radius is scope)
        are involved in path generation.
            > wmc: work memory capacity (WMC), denotes the number of the states in path.
            > precision: if the distance between two points is less than precision,
        we think they are at same location.
            > order: use for B-spline interpolation, order must be >=4.
        > times: for recording how many path have been generated.
        > top: the largest index of the points involved in path generation on route.
        > low: the smallest index of the points involved in path generation on route.
        > route: an list of Point3Ds from source the goal.
        > path: an list of States from current location of vehicle to the aim.
        > aim: the last point of the path.
        > goal: the last point of the route.
        > basis: the points are involved in path generation.
        > gridmap: the GridMap class.
    """

    def __init__(self, gridmap=None):
        # type: (GridMap) -> None
        """
        :param gridmap: Gridmap class
        """
        self.top = 0
        self.low = 0
        self.times = 0
        self.basis = []
        self.path = []
        self.route = []
        self.aim = None
        self.goal = None
        self.gridmap = gridmap
        self.config = PathConfig(scope=40., wmc=4, precision=1., order=4)

    def refresh(self, route):
        # type: (List[Location]) -> None
        """
        :param route: an list of Location from source the goal.
        """
        self.route = route[:]
        self.goal = route[-1]

    def build_path(self):
        # type: () -> None
        """
        build a path from vehicle current state to the aim.
        :return: List[State]
        """
        if len(self.route) < self.config.order:
            rospy.logwarn('Route is Too Rough, Cannot Build Path')
        else:
            self.times += 1
            rospy.loginfo('Start Build {} th Path'.format(self.times))
            self.path[:] = []
            self.update_basis()
            self.yield_focuses()
            self.aim = self.path[-1]
            # logging
            rospy.loginfo(self.table)

    def update_basis(self):
        # type: () -> None
        """
        refresh the points are involved in path generation.
        :return:
        """
        # check if it needs to reset top and low
        org = self.gridmap.origin
        pts = [[pt.x, pt.y, 0] for pt in self.route]
        gaps = np.array(pts) - np.array([org.location.x, org.location.y, 0])
        norms = list(np.linalg.norm(gaps, axis=1))
        mid = norms.index(min(norms))
        if self.top < mid:
            self.top = mid
            self.low = mid

        # update top
        n = np.linalg.norm(org.location.vec - self.route[self.top].vec)
        d = self.top + 1 - self.config.order
        g = len(self.route) - 1 - self.top
        while (n < self.config.scope or d < 0) and g > 0:
            self.top += 1
            n = np.linalg.norm(org.location.vec - self.route[self.top].vec)
            d = self.top + 1 - self.config.order
            g = len(self.route) - 1 - self.top
        # update low
        self.low = self.top + 1 - self.config.order
        while self.low > 0:
            gap = np.linalg.norm(org.location.vec - self.route[self.low].vec)
            if gap >= self.config.scope:
                break
            self.low -= 1

        self.basis = self.route[self.low: self.top + 1]

    def yield_focuses(self):
        # type: () -> None
        """
        generate states to make path up.
        distance between two neighboring elements in path is roughly scope/wmc
        :return:
        """
        org = self.gridmap.origin
        pts = self.interpolate2d()
        gaps = np.array(pts) - np.array([org.location.x, org.location.y, 0])
        norms = list(np.linalg.norm(gaps, axis=1))
        where = norms.index(min(norms))
        pts[0:where] = []

        gaps = np.array(pts) - np.array([org.location.x, org.location.y, 0])
        norms = np.linalg.norm(gaps, axis=1)

        for i in range(1, self.config.wmc):
            sl_ = list(np.fabs(norms - i * self.config.scope / self.config.wmc))
            n = sl_.index(min(sl_))
            aux = n + 1 if n + 1 in range(len(pts)) else n - 1
            yaw = np.arctan2(pts[aux][1] - pts[n][1], pts[aux][0] - pts[n][0])
            self.path.append(State(location=Location(vec=pts[n]),
                                   rotation=Rotation(rpy=(0., 0., yaw)),
                                   velocity=Velocity(vec=(0., 0., 0.))))

        if len(self.path) < self.config.wmc:
            yaw = np.arctan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
            self.path.append(State(location=Location(vec=pts[-1]),
                                   rotation=Rotation(rpy=(0., 0., yaw)),
                                   velocity=Velocity(vec=(0., 0., 0.))))

    def interpolate2d(self, degree=3, delta=0.002):
        # type: (int, float) -> List[List[float]]
        """
        :param degree: param of B-spline
        :param delta: param of B-spline, number of interpolated points is 1/delta.
        :return: interpolated points
        """
        curve = BSpline.Curve()
        curve.degree = degree
        curve.delta = delta
        curve.ctrlpts = [[base.x, base.y, 0] for base in self.basis]

        remains = len(self.basis) + degree + 1 - 2 * (degree + 1)
        inters = np.array(range(1, remains + 1)) * 1.0 / (remains + 1.0)
        curve.knotvector.extend(np.zeros(degree + 1))
        curve.knotvector.extend(inters)
        curve.knotvector.extend(np.ones(degree + 1))
        return curve.evalpts

    def is_aim(self, location):
        # type: (Location) -> bool
        """
        check if the point is on same location as the aim
        :param location: Location, need to be checked point.
        :return: True, if the location is same as the aim. False, if not.
        """
        gap = np.linalg.norm(location.vec[:2] - self.aim.location.vec[0:2])
        return gap <= self.config.precision

    def is_goal(self, location):
        # type: (Location) -> bool
        """
        check if the point is on same location as the aim
        :param location: Location, need to be checked point.
        :return: True, if the location is same as the goal. False, if not.
        """
        gap = np.linalg.norm(location.vec[:2] - self.goal.vec[0:2])
        return gap <= self.config.precision

    @property
    def table(self, title='Path Build Results'):
        """
        :param title: title of table
        :return: tabulate
        """
        table = [['Path {}th State'.format(i),
                  'x@ {}, y@{}, z@ {}, yaw @{}'.format(
                      item.location.x, item.location.y,
                      item.location.z, np.degrees(item.rotation.yaw))]
                 for i, item in enumerate(self.path)]
        return tabulate(table, headers=[title, 'Value'], tablefmt='orgtbl')
