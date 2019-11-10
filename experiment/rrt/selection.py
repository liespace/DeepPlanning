#!/usr/bin/env python
"""
Module for selecting a expected Way from a Tree.
"""
from collections import namedtuple
from copy import deepcopy
from anytree import PreOrderIter
from anytree.search import findall
from numpy.linalg import norm
import numpy as np
import rospy
from tabulate import tabulate
from dtype import TreeNode, C2Go, NStatus, C2GoType, RoadNode, WStatus
from propagation import Propagator  # pylint: disable=unused-import
from vehicle import Vehicle  # pylint: disable=unused-import


class Way(object):
    """
    Represents the way selected by rrt planner.
        > way: list of RoadNodes
        > status: status of WayStatus(Enum),
        denotes the way go with acceleration or deceleration
        > speeds: list of floats, each speed for each node in way respectively
        > target_speed: a speed is expected when vehicle finished the way.
        > mcr: minimum curvature radius of way.
    """
    __slots__ = ('way', 'status', 'speeds', 'v_t', 'mcr')

    def __init__(self):
        # type: () -> None
        self.way = []
        self.speeds = []
        self.v_t = 0
        self.status = WStatus.PULL
        self.mcr = None

    def plan_speeds(self, vehicle):
        # type: (Vehicle) -> None
        """
        plan a speeds for the points in way.
        :param vehicle: Vehicle class, provides info demanded in speeds planning
        :return:
        """
        if not self.way:
            rospy.logwarn('Way is Empty, No Need to Plan Velocities')
        else:
            self.set_target_speed(vehicle=vehicle)
            self.compute_speeds_by_energy(vehicle=vehicle)
            self.put_speeds_into_way()
            # logging
            rospy.loginfo(self.table)

    def set_target_speed(self, vehicle):
        # type: (Vehicle) -> None
        """
        set target speed according to the speed setting,
        and minimum curvature radius of the way.
        :param vehicle: Vehicle class, provides info demanded in speeds planning
        :return:
        """
        if self.status is WStatus.PULL or len(self.way) is 1:
            self.v_t = 0.
            self.mcr = None
        else:
            k_s = [self.way[i].cost2go.k for i in range(1, len(self.way))]
            k_s.sort()
            max_curvature = k_s[-1]
            self.mcr = 1. / (max_curvature + 1e-10)
            max_turn_speed = np.sqrt(vehicle.vinfo.mlaa * self.mcr)
            self.v_t = sorted([max_turn_speed, vehicle.vinfo.speed])[0]
        return self.v_t

    def compute_speeds_by_energy(self, vehicle):
        # type: (Vehicle) -> None
        """
        compute speed by equally distributing the delta energy between
        source and target. we assume the points of way is equally distributed
        because the points are generate through Uniform Interpolation.
        :param vehicle: Vehicle class, provides info demanded in speeds planning
        :return:
        """
        self.speeds[:] = []
        v_0 = np.linalg.norm([vehicle.status.twist.linear.x,
                              vehicle.status.twist.linear.y])
        v_t = self.v_t
        n = len(self.way)
        if n is 1:
            self.speeds.append(v_t)
        else:
            if v_0 < 3.0 and v_t < 0.5 and len(self.way) > 2:
                m, v_m = int(np.ceil(len(self.way)/2.0)), 3.0
                self.speeds.extend([
                    np.sqrt(v_0**2 + float(i) / (m-1) * (v_m**2 - v_0**2))
                    for i in range(0, m)])
                self.speeds.extend([
                    np.sqrt(v_m**2 + float(i) / (n-m) * (v_t**2 - v_m**2))
                    for i in range(1, n-m+1)])
            else:
                self.speeds.extend([
                    np.sqrt(v_0**2 + float(i) / (n-1) * (v_t**2 - v_0**2))
                    for i in range(n)])

    def put_speeds_into_way(self):
        # type: () -> None
        """
        set computed speeds into way.
        :return:
        """
        for i, node in enumerate(self.way):
            vel = [self.speeds[i]] + [0.] * (node.state.velocity.vec.size - 1)
            vec = np.dot(node.state.rotation.rotation_matrix(), vel)
            node.state.velocity.reset(vec)

    @property
    def table(self, title='Way Build Results'):
        """
        :param title: title of table
        :return: tabulate
        """
        table = [['Way {}th State'.format(i),
                  'x@ {}, y@{}, z@ {}, yaw @{}, speed@{}'.format(
                      item.location.x, item.location.y, item.location.z,
                      np.degrees(item.rotation.yaw), self.speeds[i])]
                 for i, item in enumerate(self.way)]
        return tabulate(table, headers=[title, 'Value'], tablefmt='orgtbl')


SelectorConfig = namedtuple(
    'SelectorConfig',
    ['duration', 'density'])
"""
# duration: maximum times for selector to cyclically run.
# density: density of interpolation points, means 1/density points per meter.
"""


class WaySelector(object):
    """
    select way from a tree
        > duration: maximum times for selector to cyclically run.
        > density: density of interpolation points,
        means 1/density points per meter
        > trace: selected best TreeNodes for forming way.
        > anode: the TreeNode has been selected finally.
        > way: selected way
        > track: interpolation points based on selected TreeNodes (trace),
        all in it is collision-free.
    """
    def __init__(self, propagator=None):
        # type: (Propagator) -> None
        """
        :param propagator: Propagator class
        """
        self.trace = []
        self.track = []
        self.way = Way()
        self.is_solved = False
        self.propagator = propagator
        self.config = SelectorConfig(duration=10, density=1.)

    def search_way(self):
        # type: () -> None
        """
        search way
        """
        rospy.loginfo('Searching way')
        if not self.propagator.root:
            rospy.logwarn('Tree is Empty, No Need to Search Way')
            return
        times, self.is_solved = 0, False
        while times < self.config.duration and not self.is_solved:
            times += 1
            if times < self.config.duration * 0.8:
                self.find_best_trace()
            else:
                self.find_near_trace()
            self.adjust_trace()
            self.is_solved = self.build_track()
        self.reset_anode()
        self.build_way(times=times)
        self.way.plan_speeds(self.propagator.vehicle)
        # logging
        rospy.loginfo(self.table)

    def reset_anode(self):
        """
        reset anode of propagator
        """
        self.propagator.anode = self.anode

    @property
    def anode(self):
        """
        :return: State
        """
        return self.trace[-1] if self.trace and self.is_solved else None

    def find_near_trace(self):
        # type: () -> None
        """
        find the trace (list of TreeNode) whose last node is the nearest one
        to the aim in the Tree.
        """
        self.trace[:] = []
        space = list(PreOrderIter(self.propagator.root))
        near = sorted(space, key=lambda node: norm(
            node.state.location.vec[0:2] -
            self.propagator.director.aim.location.vec[0:2]))[0]
        trace = list(near.path)
        if not self.propagator.director.is_aim(near.state.location):
            aimer = TreeNode(name='aim_near',
                             state=deepcopy(self.propagator.director.aim),
                             c2go=C2Go(c2gtype=C2GoType.BREAK,
                                       vec=near.c2go.vec + near.c2get.vec),
                             c2get=C2Go(vec=(0., 0., 0., 0.),
                                        c2gtype=C2GoType.BREAK),
                             status=NStatus.STOP,
                             parent=near)
            trace.append(aimer)
        self.trace[:] = trace[:]

    def find_best_trace(self):
        # type: () -> None
        """
        find the trace (list of TreeNode) whose last node with the minimum cost
        to the aim in the Tree.
        """
        self.trace[:] = []
        space = findall(self.propagator.root,
                        filter_=lambda node: node.status is NStatus.FREE)
        if len(space) < 1:
            return
        best = sorted(
            space, key=lambda node: node.c2go.cost() + node.c2get.cost())[0]
        trace = list(best.path)
        if not self.propagator.director.is_aim(best.state.location):
            aimer = TreeNode(name='aim',
                             state=deepcopy(self.propagator.director.aim),
                             c2go=C2Go(vec=best.c2go.vec + best.c2get.vec,
                                       c2gtype=C2GoType.CLOSE),
                             c2get=C2Go(vec=(0, 0, 0, 0),
                                        c2gtype=C2GoType.CLOSE),
                             status=NStatus.FREE,
                             parent=best)
            trace.append(aimer)
        self.trace[:] = trace[:]

    def build_track(self):
        # type: () -> bool
        """
        generate interpolation points based on selected TreeNodes (trace),
        all in track is collision-free.
        """
        self.track[:] = []
        if len(self.trace) < 1:
            return False
        for i in range(1, len(self.trace)):
            c2go, curve = self.propagator.compute_cost(
                start=self.trace[i - 1].state,
                end=self.trace[i].state,
                ratio=self.config.density)
            if c2go.c2gtype is C2GoType.CLOSE:
                for j in range(1, len(curve)):
                    c2go, _ = self.propagator.compute_cost(
                        start=curve[j - 1], end=curve[j])
                    self.track.append(RoadNode(state=curve[j], cost2go=c2go,
                                               status=self.trace[i].status))
            else:
                if self.trace[i] not in self.trace[i-1].children:
                    self.propagator.print_tree()
                # remove node from the tree
                children = list(self.trace[i-1].children)
                children.remove(self.trace[i])
                self.trace[i-1].children = children
                self.trace[:] = self.trace[:i]
                self.trace[-1].status = NStatus.STOP
                return False
        self.track.insert(0, self.trace[0].to_road_node())
        return True

    def build_way(self, times):
        # type: (int) -> None
        """
        build way
        """
        is_solved = self.is_solved
        is_best = times < self.config.duration * 0.8
        is_goal = (self.propagator.director.is_goal(self.track[-1].state.location)
                   if self.track else False)
        self.way.way[:] = self.track[:]
        if not is_solved or not is_best or is_goal:
            self.way.status = WStatus.PULL
        else:
            self.way.status = WStatus.PUSH

    def adjust_trace(self):
        # type: () -> None
        """
        re-propagation for cover the movement of the vehicle when
        propagating tree.
        """
        if not self.trace or len(self.trace) <= 1:
            return
        root_state = deepcopy(self.propagator.gridmap.origin)
        child, _ = self.propagator.nearest_child_of(whom=root_state,
                                                    space=self.trace)
        if child:
            if child.name is self.trace[0].name:
                self.trace[0].state = root_state
            else:
                where = self.trace.index(child)
                self.trace[:where] = []
                self.propagator.root.state = root_state
                self.trace.insert(0, self.propagator.root)

    @property
    def table(self, title='Way Search Results'):
        """
        :param title: title of table
        :return: tabulate
        """
        table = [['Is Solved', self.is_solved],
                 ['Way Status', self.way.status],
                 ['Way Size', len(self.way.way)],
                 ['Track Size', len(self.track)],
                 ['Trace Size', len(self.trace)]]
        return tabulate(table, headers=[title, 'Value'], tablefmt='orgtbl')


class DWWaySelector(WaySelector):
    def __init__(self, propagator=None):
        super(DWWaySelector, self).__init__(propagator=propagator)

    def search_way(self):
        # type: () -> None
        """
        search way
        """
        rospy.loginfo('Searching way')
        if not self.propagator.root:
            rospy.logwarn('Tree is Empty, No Need to Search Way')
            return
        times, self.is_solved = 0, False
        while times < self.config.duration and not self.is_solved:
            times += 1
            self.find_best_trace()
            self.is_solved = self.build_track()
        self.reset_anode()
        self.build_way(times=times)
        # self.way.plan_speeds(self.propagator.vehicle)
        # logging
        rospy.loginfo(self.table)
