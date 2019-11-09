#!/usr/bin/env python
"""
Module of Tree Propagation
"""
from __future__ import print_function
import random
from collections import namedtuple
from copy import deepcopy
import logging
from typing import Tuple, List, Union  # pylint: disable=unused-import
import matplotlib.pyplot as plt
from tabulate import tabulate
from anytree import PreOrderIter, RenderTree
import dubins
import reeds_shepp
import numpy as np
from map import GridMap  # pylint: disable=unused-import
from task import Director  # pylint: disable=unused-import
from vehicle import Vehicle  # pylint: disable=unused-import
from sampling import GBSE2Sampler, DWGBSE2Sampler, SamplerConfig, DWSamplerConfig  # pylint: disable=unused-import
from dtype import TreeNode, State, C2Go, C2GoType
from dtype import NStatus, Location, Rotation, Velocity


PropagatorConfig = namedtuple(
    'PropagatorConfig',
    ['duration', 'density', 'precision'])
"""
# duration: maximum times for propagator to cyclically run.
# density: density of interpolation points, means 1/density points per meter.
# precision: if the distance between two points is less than precision,
  we think they are at same location.
"""


class Propagator(object):  # pylint: disable=too-many-instance-attributes
    """
    Propagate Tree
        > root: root node of tree. represent the all tree in practice.
        > seq: sequence number for each node.
        > config: ['duration', 'density', 'precision']
    """

    def __init__(self, director=None, vehicle=None, gridmap=None, sampler=None):
        # type: (Director, Vehicle, GridMap, Union[GBSE2Sampler, DWGBSE2Sampler]) -> None
        """
        :param director: Director class
        :param vehicle: Vehicle class
        :param gridmap: Gridmap class
        :param sampler: GBSE2Sampler class
        """
        self.seq = 0
        self.new = None
        self.root = None
        self.anode = None
        self.gridmap = gridmap
        self.sampler = sampler
        self.vehicle = vehicle
        self.director = director
        self.config = PropagatorConfig(duration=100, density=0.3, precision=0.2)

    def propagate(self):
        # type: () -> None
        """
        Propagate the tree
        """
        if not self.director.path:
            logging.warning('Path is Empty, No Need to Propagation')
        else:
            logging.info('Propagating tree')
            self.refresh()
            times, unlucky, unadded, repeat = -1, 0, 0, 0
            while times < self.config.duration:
                times += 1
                if not self.sampling(base=self._get_base(times=times)):
                    unlucky += 1
                    continue
                if not self.is_necessary(who=self.sampler.sample):
                    repeat += 1
                    continue
                if not self.build_edge(parent=self.parent_of_new()):
                    unadded += 1
            # self.render_tree()
            logging.info(self.table([times, unlucky, repeat, unadded]))

    def _get_base(self, times=0):
        """
        :param times: a count of sampling times
        :return: a biased base for sampling
        """
        return self.director.path[times % len(self.director.path)]

    def sampling(self, base):
        """
        :param base: biased base
        :return: if sampled a sample or not
        """
        return self.sampler.sampling(base=base)

    def refresh(self):
        # type: () -> None
        """
        Refresh the tree
        """
        logging.info('Refreshing the Tree.')
        if not self.root or not self.anode or self.anode.name is self.root.name:
            self.init()
        else:
            self.trim()
        self._update_c2get()
        logging.info('Refreshed, Tree Size is %d', self.tree_size)

    @property
    def tree_size(self):
        """
        :return: size of the tree
        """
        return len(list(PreOrderIter(self.root)))

    def init(self):
        # type: () -> None
        """
        initialize the tree
        """
        logging.info('Initializing the tree.')
        self.root = TreeNode(name='root',
                             state=deepcopy(self.gridmap.origin),
                             c2go=C2Go(vec=(0., 0., 0., 0.)),
                             c2get=C2Go(),
                             status=NStatus.FREE)

    def trim(self):
        # type: () -> None
        """
        Trim the tree
        """
        logging.info('Trimming tree')
        if not self.anode or self.anode.name is self.root.name:
            self.root.state = deepcopy(self.gridmap.origin)
            self.root.children = []
            logging.info("keep no branch")
        else:
            path = list(self.anode.path)  # path:root->anode, include root and anode.
            self.root.state = deepcopy(self.gridmap.origin)
            head, c2head = self.nearest_child_of(whom=self.root.state)
            if not c2head:
                self.root.children = []
                logging.info("all branches are trimmed")
            else:
                for node in path[1:]:  # root isn't involved
                    node.parent.children = [node]
                self.root.children = [head]
                head.parent = self.root
                logging.info("keep branches of node@ %s", head.name)
                delta = c2head.vec - head.c2go.vec
                for node in PreOrderIter(head):
                    node.c2go.reset_vec(node.c2go.vec + delta)

    def _update_c2get(self):
        # type: () -> None
        """
        Update cost to aim (cost to get, c2get) of nodes in tree.
        """
        logging.info('Updating Cost to Get')
        for n in PreOrderIter(self.root):
            n.c2get, _ = self.compute_cost(start=n.state, end=self.director.aim)

    def compute_cost(self, start, end, ratio=2.0):
        # type: (State, State, float) -> Tuple[C2Go, List[State]]
        """
        build curve and make up the Cost-To-GO.
        """
        c2go, curve = self.build_curve(start=start, end=end, ratio=ratio)
        drivable = self.is_curve_drivable(c2go)
        if not drivable:
            c2go.c2gtype = C2GoType.BREAK
            return c2go, curve
        free, grade = self.is_curve_free(curve)
        if not free:
            c2go.c2gtype = C2GoType.BREAK
            return c2go, curve
        c2go.grade = grade
        c2go.reset_vec(vec=np.fabs(c2go.vec))
        c2go.c2gtype = C2GoType.CLOSE
        return c2go, curve

    def nearest_child_of(self, whom, space=None):
        # type: (State, List[TreeNode]) ->  Tuple[TreeNode, C2Go]
        """
        Find the nearest child of a state.
        """
        min_cost, child, c2child = np.inf, None, None
        space = space if space else PreOrderIter(self.root)
        for node in space:
            c2node, _ = self.compute_cost(start=whom, end=node.state)
            if c2node.c2gtype is C2GoType.CLOSE:
                if c2node.cost() < min_cost:
                    min_cost = c2node.cost()
                    child, c2child = node, c2node
        return child, c2child

    def nearest_parent_of(self, whom, space=None):
        # type: (State, List[TreeNode]) ->  Tuple[TreeNode, C2Go]
        """
         Find the nearest parent of a state.
        """
        min_cost, parent, c2parent = np.inf, None, None
        space = space if space else PreOrderIter(self.root)
        for node in space:
            c2node, _ = self.compute_cost(start=node.state, end=whom)
            if c2node.c2gtype is C2GoType.CLOSE:
                if c2node.cost() < min_cost:
                    min_cost = c2node.cost()
                    parent, c2parent = node, c2node
        return parent, c2parent

    def build_curve(self, start, end, ratio=2.0):
        # type: (State, State, float) -> Tuple[C2Go, List[State]]
        """
        wrap function for curve building, default curve type is clothoid
        """
        return self.build_dubins(start=start, end=end,
                                 ratio=ratio, vehicle=self.vehicle)

    # This implement have a problem: demanded npts is not always equal to outputs
    # @staticmethod
    # def build_spiral2d(start, end, ratio=2.0):
    #     # type: (State, State, float) -> Tuple[C2Go, List[State]]
    #     """
    #     Build euler spiral, or called clothoid, the returned curve includes both
    #     start and end.
    #     """
    #     q_0 = np.array([start.location.x, start.location.y, start.rotation.y])
    #     q_1 = np.array([end.location.x, end.location.y, end.rotation.y])
    #     k, dk, l, _ = build_clothoid(q_0[0], q_0[1], q_0[2],
    #                                  q_1[0], q_1[1], q_1[2])
    #     xy_ = points_on_clothoid(q_0[0], q_0[1], q_0[2],
    #                              k, dk, l, int(l*ratio)+1)
    #     npts = len(xy_[0])
    #     t = np.linspace(0, l, npts)
    #     rpy = t * k + dk / 2. * np.power(t, 2.) + q_0[2]
    #     rpy = np.array([np.zeros(npts), np.zeros(npts), rpy]).transpose()
    #     xy_ = np.array([xy_[0], xy_[1], np.zeros(npts)]).transpose()
    #     return (C2Go(vec=[l, k, dk, np.linalg.norm(q_0[0:2] - q_1[0:2])]),
    #             [State(Location(vec=xy_[i]), Rotation(rpy=rpy[i]), Velocity())
    #              for i in range(npts)])

    @staticmethod
    def build_dubins(start, end, vehicle, ratio=2.0):
        # type: (State, State, Vehicle, float) -> Tuple[C2Go, List[State]]
        """
        build dubins
        """
        q_0 = np.array((start.location.x, start.location.y, start.rotation.y))
        q_1 = np.array((end.location.x, end.location.y, end.rotation.y))
        p = dubins.shortest_path(q_0, q_1, vehicle.vinfo.attrs.mtr)
        if p.path_length() > 30:
            return C2Go(), []
        configurations, _ = p.sample_many(1.0 / ratio)
        configurations.append(tuple(q_1))  # include the end point
        npts = len(configurations)
        cfg = np.array(configurations).transpose()
        xyz = np.stack((cfg[0], cfg[1], np.zeros(npts))).transpose()
        rpy = np.stack((np.zeros(npts), np.zeros(npts), cfg[2])).transpose()
        return (C2Go(vec=[p.path_length(), 0., 0.,
                          np.linalg.norm(q_0[0:2] - q_1[0:2])]),
                [State(Location(xyz[i]), Rotation(rpy[i]), Velocity())
                 for i in range(npts)])

    @staticmethod
    def build_reeds_shepp(start, end, vehicle, ratio=2.0):
        # type: (State, State, Vehicle, float) -> Tuple[C2Go, List[State]]
        """
        build Reeds Shepp Curve
        """
        q_0 = np.array((start.location.x, start.location.y, start.rotation.y))
        q_1 = np.array((end.location.x, end.location.y, end.rotation.y))
        configurations = reeds_shepp.path_sample(
            q_0, q_1, vehicle.vinfo.attrs.mtr, 1.0 / ratio)
        configurations.append(tuple(q_1))  # include the end point
        npts = len(configurations)
        cfg = np.array(configurations).transpose()
        xyz = np.stack((cfg[0], cfg[1], np.zeros(npts))).transpose()
        rpy = np.stack((np.zeros(npts), np.zeros(npts), cfg[2])).transpose()
        dist = reeds_shepp.path_length(q_0, q_1, vehicle.vinfo.attrs.mtr)
        return (C2Go(vec=[dist, 0., 0.,
                          np.linalg.norm(q_0[0:2] - q_1[0:2])]),
                [State(Location(xyz[i]), Rotation(rpy[i]), Velocity())
                 for i in range(npts)])

    def is_curve_free(self, curve):
        # type: (List[State]) -> Tuple[bool, int]
        """
        check if a curve is collision-free or not.
        :param curve: list of State
        :return:
        """
        i, grade = False, 0
        for state in curve:
            i, g = self.gridmap.is_free(state=state)
            grade += g
            if not i:
                return i, np.inf
        return i, grade

    def is_curve_drivable(self, c2go):  # type: (C2Go) -> bool
        """
        check if the curvature of a curve is admissible.
        :param c2go: C2Go class
        :return: True, if admissible, False, if not
        """
        # [k, k + dk * l]
        k_s = [c2go.vec[1], c2go.vec[1] + c2go.vec[0] * c2go.vec[2]]
        return self.vehicle.is_turnable(k_s=k_s)

    def build_edge(self, parent):
        # type: (TreeNode) -> bool
        """
        build edge
        """
        logging.debug('Building Edge')
        if not parent:
            self.new = None
            return False
        c2go, _ = self.compute_cost(start=parent.state, end=self.sampler.sample)
        if c2go.c2gtype is C2GoType.BREAK:
            self.new = None
            return False
        c2get, _ = self.compute_cost(start=self.sampler.sample, end=self.director.aim)
        self.new = TreeNode(name='{}_{}'.format('node', self._sequence),
                            state=deepcopy(self.sampler.sample),
                            c2get=c2get,
                            c2go=C2Go(vec=(parent.c2go.vec + c2go.vec),
                                      grade=parent.c2go.grade + c2go.grade),
                            status=(NStatus.FREE if c2get.c2gtype
                                    is C2GoType.CLOSE else NStatus.STOP),
                            parent=parent)
        return True

    def rewiring(self, radius=5.0, ratio=2.0):
        """
        rewiring, two steps:
        A.  find neighbors with distance < radius to the new.
        B.  if go to a neighbor of neighbors from the new has the less cost,
            make the new become the parent of that neighbor
        :param radius: tree-nodes in the circle region with this radius
        will be seen as a member needed to be checked.
        :param ratio: for curve interpolation.
        """
        logging.debug('Rewiring')
        neighbors = []
        for node in PreOrderIter(self.root):
            if self.new.state.location.distance(node.state.location) < radius:
                neighbors.append(node)
        logging.debug('neighbors(%.1f&%.1f) number: %d', radius, ratio, len(neighbors))
        for neighbor in neighbors:
            c2go, _ = self.compute_cost(start=self.new.state,
                                        end=neighbor.state, ratio=ratio)
            if c2go.c2gtype is C2GoType.CLOSE:
                if self.new.c2go.cost() + c2go.cost() < neighbor.c2go.cost():
                    logging.info('before: %.2f, after %.2f',
                                 neighbor.c2go.cost(),
                                 self.new.c2go.cost() + c2go.cost())
                    neighbor.parent = self.new
                    delta = self.new.c2go.vec + c2go.vec - neighbor.c2go.vec
                    for node in PreOrderIter(neighbor):
                        node.c2go.reset_vec(node.c2go.vec + delta)

    def parent_of_new(self, space=None):
        # type: (List[TreeNode]) -> TreeNode
        """
        find parent with heuristic
        """
        minimum, parent = np.inf, None
        space = space if space else PreOrderIter(self.root)
        for node in space:
            heu = self.heuristic(start=node.state, end=self.sampler.sample)
            if heu and heu < minimum:
                minimum = heu
                parent = node
        return parent

    def heuristic(self, start, end):
        # type: (State, State) -> Union[float, None]
        """
        calculate the heuristic from start to end
        :param start: start State
        :param end: end State
        :return: heuristic
        """
        q_0 = np.array((start.location.x, start.location.y, start.rotation.y))
        q_1 = np.array((end.location.x, end.location.y, end.rotation.y))
        path = dubins.shortest_path(q_0, q_1, self.vehicle.vinfo.attrs.mtr)
        return path.path_length() if path else None

    def is_necessary(self, who=None, space=None):
        # type: (State, List[TreeNode]) -> bool
        """
        check if whom is necessary to be added.
        if who has distance to any node in space less than the precision, False.
        """
        space = space if space else PreOrderIter(self.root)
        for node in space:
            norm = np.linalg.norm(who.location.vec[0:2] -
                                  node.state.location.vec[0:2])
            theta = np.fabs(who.rotation.y - node.state.rotation.y)
            if norm < self.config.precision and theta < self.config.precision:
                return False
        return True

    @property
    def _sequence(self):
        # type: () -> int
        """
        generate the sequence number of the nodes
        """
        self.seq += 1
        return self.seq

    def print_tree(self):
        """
        comprehension print of the tree.
        """
        for pre, _, node in RenderTree(self.root):
            treestr = u"%s%s [(%.2f %.2f) (%d)]" % (pre, node.name,
                                                    node.state.location.x,
                                                    node.state.location.y,
                                                    node.status.value)
            print(treestr.ljust(8))

    def table(self, params, title='Tree Propagation Results'):
        """
        :param params: parameters those required to show up.
        :param title: title of table
        :return: tabulate
        """
        table = [['Tree Size', len(list(PreOrderIter(self.root)))],
                 ['Times', params[0]],
                 ['Un-lucky Times', params[1] / params[0]],
                 ['Repeated Times', params[2] / params[0]],
                 ['Un-added Times', params[3] / params[0]]]
        return tabulate(table, headers=[title, 'Value'], tablefmt='orgtbl')


class BiPropagator(Propagator):
    """
    bi-rrt, subclass of Propagator, extension:
    > minor: the other root
    """
    def __init__(self, director=None, vehicle=None, gridmap=None, sampler=None):
        super(BiPropagator, self).__init__(director, vehicle, gridmap, sampler)
        self.path = []
        self.main = None
        self.minor = None
        self.config = PropagatorConfig(duration=150, density=2.0, precision=0.1)
        self.sampler.config = SamplerConfig(r_mean=2.0, r_sigma=0.5,
                                            t_mean=0., t_sigma=np.pi / 4.,
                                            h_mean=0., h_sigma=np.pi / 6.)
        self.is_first = True
        self.extend = 0

    def propagate(self):  # type: () -> None
        """
        :return:
        """
        logging.debug('Propagating tree')
        if not self.director.aim:
            logging.warning('Aim is Empty, No Need to Propagation')
        else:
            if self.is_first:
                self.refresh()
            times, unlucky, unadded, repeat, is_over = -1, 0, 0, 0, False
            while (times < (self.config.duration + self.extend)
                   and not (is_over and self.is_first)):
                times += 1
                self._switch(times)
                if not self.sampling(base=self._get_base(times=times)):
                    unlucky += 1
                    continue
                if not self.is_necessary(who=self.sampler.sample):
                    repeat += 1
                    continue
                if not self.build_edge(parent=self.parent_of_new()):
                    unadded += 1
                    continue
                # self.rewiring(radius=5.0, ratio=4.0)
                if self._is_over():
                    is_over = True
                    self._extract()
            self.sort_path()
            # logging.info(self.table([times, unlucky, repeat, unadded, is_over]))

    def build_curve(self, start, end, ratio=2.0):
        # type: (State, State, float) -> Tuple[C2Go, List[State]]
        """
        wrap function for curve building, default curve type is clothoid
        """
        return self.build_reeds_shepp(
            start=start, end=end, ratio=ratio, vehicle=self.vehicle)

    def _switch(self, times):
        """
        switch the root between main and minor
        :param times: propagation times
        :return:
        """
        self.root = self.main if not times % 2 else self.minor

    def _is_over(self):  # type: ()->bool
        """
        check if there is a node in the other tree can be got from the new.
        """
        aux = self.minor if self.root.name == self.main.name else self.main
        space = list(PreOrderIter(aux))
        c2gos = []
        for n in space:
            if n.name == self.new.name:
                continue
            c2go, _ = self.compute_cost(start=self.new.state, end=n.state)
            if c2go.c2gtype is C2GoType.CLOSE:
                c2gos.append([c2go.cost(), n])
        if not c2gos:
            return False
        self.anode = sorted(c2gos)[0][-1]  # type: TreeNode
        return True

    def _extract(self):  # type: ()->None
        """
        extract the path(root->itself) from main the minor, use anode and new.
        """
        if self.root.name == self.main.name:
            forward = [n.replica for n in self.new.path]
            backward = [n.replica for n in self.anode.path]
        else:
            forward = [n.replica for n in self.anode.path]
            backward = [n.replica for n in self.new.path]
        parent = None
        for n in forward:
            n.parent = parent
            parent = n
        backward[-1].parent = parent
        parent = backward[-1]
        for n in reversed(backward[:-1]):
            n.parent = parent
            parent = n
        self._add_path(forward[0])

    def _add_path(self, source):  # type: (TreeNode) -> None
        """add path and re-compute the c2go, only store the last node"""
        for node in PreOrderIter(source):  # type: TreeNode
            if not node.parent:
                continue
            c2go, _ = self.build_curve(
                start=node.parent.state, end=node.state, ratio=0.01)
            node.c2go.reset_vec(vec=node.parent.c2go.vec + np.fabs(c2go.vec))
        source.name = 'path'
        self._refine(source.leaves[0])
        self.path.append(source.leaves[0])

    def sort_path(self):
        if self.path:
            self.path.sort(key=lambda node: node.c2go.cost())

    def _refine(self, target):  # type: (TreeNode) -> None
        logging.debug('Refining')
        child = target
        while child.parent and child.parent.parent:
            papa = child.parent.parent
            c2go, _ = self.compute_cost(start=papa.state, end=child.state)
            if c2go.c2gtype is C2GoType.CLOSE:
                if papa.c2go.cost() + c2go.cost() < child.c2go.cost():
                    logging.debug('before: %.2f, after %.2f',
                                  child.c2go.cost(),
                                  papa.c2go.cost() + c2go.cost())
                    child.parent.parent = None
                    child.parent = papa
                    child.c2go.reset_vec(vec=papa.c2go.vec + c2go.vec)
                    child = papa
                    continue
            child = child.parent

    def _get_base(self, times=0):
        """
        get biased base for sampling
        :param times:
        :return:
        """
        return random.choice(list(PreOrderIter(self.root))).state

    def refresh(self):  # type: () -> None
        """
        refresh the tree status
        :return:
        """
        self.init()

    def init(self):  # type: () -> None
        """
        initialize the tree
        """
        logging.debug('Initializing the tree.')
        self.main = TreeNode(name='main',
                             state=deepcopy(self.gridmap.origin),
                             c2go=C2Go(vec=(0., 0., 0., 0.)),
                             c2get=C2Go(),
                             status=NStatus.FREE)
        self.minor = TreeNode(name='minor',
                              state=deepcopy(self.director.aim),
                              c2go=C2Go(vec=(0., 0., 0., 0.)),
                              c2get=C2Go(),
                              status=NStatus.FREE)

    def plot(self, filepath):
        """
        plot tree, path and grid map
        :param filepath: image saved filepath
        :return:
        """
        plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')
        self.plot_grid()
        self.plot_space(space=list(PreOrderIter(self.main)), color='b')
        self.plot_space(space=list(PreOrderIter(self.minor)), color='g')
        # self.plot_space(space=list(PreOrderIter(self.root)), color='r')
        self.plot_path(color='r')
        plt.show()
        # plt.savefig('{}.eps'.format(filepath), bbox_inches='tight', format='eps')

    def plot_grid(self):
        """plot grid map"""
        row, col = self.gridmap.grid.shape[0], self.gridmap.grid.shape[1]
        u = np.array(range(row)).repeat(col)
        v = np.array(range(col)*row)
        uv = np.array([u, v, np.ones_like(u)])
        xy = np.dot(np.linalg.inv(self.gridmap.xy2uv), uv)
        data = {'x': xy[0, :], 'y': xy[1, :],
                'c': np.array(self.gridmap.grid).flatten() - 1}
        plt.scatter(x='x', y='y', c='c', data=data, s=30., marker="s")

    def plot_path(self, color='r', width=1.0):
        """plot path"""
        if not self.path:
            return
        leaf = self.path[0]
        path = list(PreOrderIter(leaf.root))
        for i, _ in enumerate(path):
            if i == 0:
                continue
            if path[i].c2go.reverse:
                _, curve = self.compute_cost(start=path[i].state, end=path[i-1].state, ratio=4.0)
            else:
                _, curve = self.compute_cost(start=path[i-1].state, end=path[i].state, ratio=4.0)
            way = [[s.location.x, s.location.y, 0, 1] for s in curve]
            way = np.array(way).transpose()
            way = np.dot(self.gridmap.origin.transformation(inv=True), way)
            plt.plot(way[0, :], way[1, :],
                     color=color, linewidth=width, linestyle='-', )

    def plot_space(self, space, color='b'):
        """plot space"""
        for i, n in enumerate(space):
            st0 = self.gridmap.origin.transform(n.state)
            pt0 = (st0.location.x, st0.location.y, st0.rotation.y)
            if i == 0:
                self.plot_poly(se2=pt0, color=color, length=0.1)
                self.plot_circle(pt0=pt0, color=color, radius=0.02)
            for child in n.children:
                st1 = self.gridmap.origin.transform(child.state)
                pt1 = (st1.location.x, st1.location.y, st1.rotation.y)
                self.plot_arrow(se2=pt1, color=color, length=0.1)
                self.plot_circle(pt0=pt1, color=color, radius=0.02)
                self.plot_line(pt0=pt0, pt1=pt1, color=color, width=0.5)

    @staticmethod
    def plot_poly(se2=(0., 0., 0.), color='g', length=5., nvs=3):
        """plot poly"""
        x, y, theta = se2[0], se2[1], se2[2]
        gap = 2 * np.pi / nvs
        vertices = []
        for i in range(nvs):
            angle = i * gap + theta
            size = length if i > 0 else length * 3
            vertex = [x + size * np.cos(angle), y + size * np.sin(angle)]
            vertices.append(vertex)
        poly = plt.Polygon(np.array(vertices), color=color, closed=True)
        plt.gca().add_patch(poly)

    @staticmethod
    def plot_arrow(se2=(0., 0., 0.), color='k', length=1.):
        """plot arrow"""
        x, y, theta = se2[0], se2[1], se2[2]
        dx, dy = length * np.cos(theta), length * np.sin(theta)
        plt.arrow(x=x, y=y, dx=dx, dy=dy, color=color,
                  head_width=0.35*length, head_length=0.8*length, shape='full')

    @staticmethod
    def plot_line(pt0, pt1, color='k', width=2.):
        """plot line"""
        x_1, y_1, x_2, y_2 = pt0[0], pt0[1], pt1[0], pt1[1]
        plt.plot([x_1, x_2], [y_1, y_2],
                 color=color, linewidth=width, linestyle='-',)

    @staticmethod
    def plot_circle(pt0, color='b', radius=0.5):
        """plot circle"""
        x, y = pt0[0], pt0[1]
        circle = plt.Circle(xy=(x, y), radius=radius, color=color)
        plt.gca().add_patch(circle)

    def table(self, params, title='Tree Propagation Results'):
        """
        :param params: parameters those required to show up.
        :param title: title of table
        :return: tabulate
        """
        table = [['Main-Tree Size', len(list(PreOrderIter(self.main)))],
                 ['Minor-Tree Size', len(list(PreOrderIter(self.minor)))],
                 ['Times', params[0]],
                 ['Is Over', params[4]],
                 ['Un-lucky Times', 1. * params[1] / (params[0] + 1e-10)],
                 ['Repeated Times', 1. * params[2] / (params[0] + 1e-10)],
                 ['Un-added Times', 1. * params[3] / (params[0] + 1e-10)]]
        return tabulate(table, headers=[title, 'Value'], tablefmt='orgtbl')


class DWAPropagator(BiPropagator):
    def __init__(self, director=None, vehicle=None, gridmap=None, sampler=None):
        # type: (Director, Vehicle, GridMap, DWGBSE2Sampler) -> None
        super(DWAPropagator, self).__init__(director, vehicle, gridmap, sampler)
        self.config = PropagatorConfig(duration=150, density=2.0, precision=0.1)
        self.sampler.config = DWSamplerConfig(x_mean=0.518, x_sigma=2.102,
                                              y_mean=-0.005, y_sigma=1.917,
                                              t_mean=-0.011, t_sigma=0.394)

    def propagate(self):
        # type: () -> None
        """
        Propagate the tree
        """
        if not self.director.path:
            logging.warning('Path is Empty, No Need to Propagation')
        elif len(self.director.path) == 1:
            print('No interchanging point')
            self.refresh()
        else:
            logging.info('Propagating tree')
            if self.is_first:
                self.refresh()
            times, unlucky, unadded, repeat, is_over = -1, 0, 0, 0, False
            while (times < (self.config.duration + self.extend)
                   and not (is_over and self.is_first)):
                if self._is_over() and self.is_first:
                    is_over = True
                    continue
                times += 1
                if not self.sampling(base=self._get_base(times=times)):
                    unlucky += 1
                    continue
                if not self.is_necessary(who=self.sampler.sample):
                    repeat += 1
                    continue
                if not self.build_edge(parent=self.parent_of_new()):
                    unadded += 1
                    continue
                # self.rewiring(radius=5.0, ratio=4.0)
            # self.render_tree()
            logging.info(super(BiPropagator, self).table(
                [times, unlucky, repeat, unadded]))

    def _get_base(self, times=0):
        if len(self.director.path) <= 2:
            return self.director.path[0]
        else:
            return self.director.path[times % (len(self.director.path) - 1)]

    def _is_over(self):  # type: ()->bool
        for node in PreOrderIter(self.root):
            if node.c2get.c2gtype is C2GoType.CLOSE:
                return True
        return False

    def _add_reference_to_tree(self):  # type: ()-> Union[None, TreeNode]
        last, c2gos = self.root.state, []
        for p in self.director.path:
            c2go, _ = self.compute_cost(start=last, end=p)
            if c2go.c2gtype is C2GoType.BREAK:
                return None
            c2gos.append(c2go)
            last = p
        path, parent = [], self.root
        for i, p in enumerate(self.director.path):
            node = TreeNode(name='path',
                            state=p,
                            c2go=c2gos[i],
                            c2get=C2Go(),
                            status=NStatus.FREE)
            node.parent = parent
            path.append(node)
            parent = node
        return path[-1]

    def refresh(self):  # type: () -> None
        super(BiPropagator, self).init()
        self._add_reference_to_tree()
        self._update_c2get()

    def plot(self, filepath):
        """
        plot tree, path and grid map
        :param filepath: image saved filepath
        :return:
        """
        plt.figure()
        plt.gca().set_aspect('equal', adjustable='box')
        self.plot_grid()
        self.plot_space(space=list(PreOrderIter(self.root)), color='b')
        # self.plot_space(space=list(PreOrderIter(self.root)), color='r')
        self.plot_path(color='r')
        plt.show()
        # plt.savefig('{}.eps'.format(filepath), bbox_inches='tight', format='eps')
