from typing import List, Tuple, Optional, Any
import time
import logging
import numba
import numpy as np
import cv2
import reeds_shepp
from debugger import Debugger


class RRTStar(object):
    def __init__(self):
        self.debug = False
        self.vertices = None  # type: Optional[List[RRTStar.StateNode]]
        self.root = None  # type: Optional[RRTStar.StateNode]
        self.gain = None  # type: Optional[RRTStar.StateNode]
        self.x_best = None  # type: Optional[RRTStar.StateNode]
        self.epsilon = 1e-6

        self.check_res = 0.3  # type: Optional[float]
        self.check_poly = None  # type: Optional[np.ndarray]
        self.exist_res = 0.1
        self.maximum_curvature = 0.2  # type: Optional[float]

        self.start = None  # type: Optional[RRTStar.StateNode]
        self.goal = None  # type: Optional[RRTStar.StateNode]
        self.grid_map = None  # type: Optional[np.ndarray]
        self.grid_res = None  # type: Optional[float]
        self.grid_ori = None  # type: Optional[RRTStar.StateNode]
        self.obstacle = None  # type: Optional[int]
        self.heuristic = None  # type: Optional[List[(Tuple[float], Tuple[float], Tuple[float])]]

    def set_vehicle(self, check_poly, check_res, maximum_curvature):
        # type: (np.ndarray, float, float) -> RRTStar
        """
        set parameter of the vehicle
        :param check_poly: contour of the vehicle for collision check. cv2.contour.
        :param check_res: the resolution of curve interpolation.
        :param maximum_curvature: equal to 1/minimum_turning_radius of the vehicle.
        """
        self.check_poly, self.check_res, self.maximum_curvature = check_poly, check_res, maximum_curvature
        return self

    def preset(self, start, goal, grid_map, grid_res, grid_ori, obstacle, heuristic):
        # type: (StateNode, StateNode, np.ndarray, float, StateNode, int, Any) -> RRTStar
        """
        initialize the parameters for planning: Start State, Goal State and other needs.
        :param start: the start state.
        :param goal: the goal state.
        :param grid_map: occupancy grid map.
        :param grid_res: resolution of grid map.
        :param grid_ori: the center point of the occupancy grid map.
        :param obstacle: the value of pixels of the obstacles region.
        :param heuristic: [(state, biasing)], sampling heuristic path.
            state: state (x_o, y_o, a_o) of the point of the path.
            biasing = (x_mu, x_sigma), (y_mu, y_sigma), (a_mu, a_sigma).
        :return: RRTStar object
        """
        self.grid_map, self.grid_res, self.grid_ori, self.obstacle = grid_map, grid_res, grid_ori, obstacle
        self.heuristic = heuristic
        self.start, self.goal = start, goal
        self.start.g, self.start.hl = 0, self.cost(start, goal)
        self.start.hu = self.start.hl if self.collision_free(start, goal) else np.inf
        self.start.fl, self.start.fu = self.start.g + self.start.hl, self.start.g + self.start.hu
        self.root, self.gain = self.start, self.goal
        self.vertices, self.x_best = [self.root], self.root
        return self

    def planning(self, times, repeat=10, optimize=False, debug=False):
        """main flow."""
        self.debug = debug
        past = time.time()
        for i in range(times):
            x_new = self.sample_free(i, repeat)
            x_nearest = self.nearest(x_new) if not optimize else self.least(x_new)
            if x_nearest and self.benefit(x_new) and self.collision_free(x_nearest, x_new):
                self.attach(x_nearest, x_new)
                self.rewire(x_new)
                self.x_best = self.best()
                self.branch_and_bound()
            Debugger().debug_planned_path(self, i, switch=self.debug)
            Debugger().debug_planning_hist(self, i, (time.time() - past), switch=True)

    def branch_and_bound(self, space=None):
        def out(x):
            vertices.remove(x)
            x.remove()
        vertices = space if space else self.vertices
        vs = filter(lambda x: x.fl > self.x_best.fu + self.epsilon, vertices)
        map(out, vs)
        Debugger().debug_branch_and_bound(vs, switch=self.debug)

    def sample_free(self, n, repeat=10, default=((2., .5), (0., np.pi / 4.), (0, np.pi / 6.))):
        """sample a state from free configuration space."""

        def is_free(state):
            contour = self.transform(self.check_poly, state[0], state[1], state[2])
            contour = np.floor(contour / self.grid_res + self.grid_map.shape[0] / 2.).astype(int)
            mask = np.zeros_like(self.grid_map, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            result = np.bitwise_and(mask, self.grid_map)
            Debugger().breaker('sample free: {}'.format(np.all(result < self.obstacle)), self.debug)
            return np.all(result < self.obstacle)

        def exist(state):
            def key(y):
                dxy = np.fabs(y.state[:-1] - s[:-1])
                da = ((y.state[-1] + np.pi) % (2 * np.pi) - np.pi) - ((s[-1] + np.pi) % (2 * np.pi) - np.pi)
                return dxy[0] < self.exist_res and dxy[1] < self.exist_res and da < self.exist_res
            s = np.array(state)
            result = filter(key, self.vertices)
            Debugger.breaker('sample exclusive: {}'.format(result == []), switch=self.debug)
            return result

        def emerge():
            if self.heuristic:
                i = n % len(self.heuristic)
                state, biasing = self.heuristic[i]
                rand = [state[0], state[1], state[2]]  # [x_o, y_o, a_o]
                (x_mu, x_sigma), (y_mu, y_sigma), (a_mu, a_sigma) = biasing
                rand[0] += np.random.normal(x_mu, x_sigma)
                rand[1] += np.random.normal(y_mu, y_sigma)
                rand[2] += np.random.normal(a_mu, a_sigma)
                return rand
            else:
                Debugger().debug_no_heuristic(vertex.state, default, self.debug)
                rand = [vertex.state[0], vertex.state[1], vertex.state[2]]
                (r_mu, r_sigma), (t_mu, t_sigma), (a_mu, a_sigma) = default
                r, theta = np.random.normal(r_mu, r_sigma), np.random.normal(t_mu, t_sigma) + rand[2]
                rand[0] += r * np.cos(theta)
                rand[1] += r * np.sin(theta)
                rand[2] += np.random.normal(a_mu, a_sigma)
                return rand

        vertex = np.random.choice(self.vertices)
        for i in range(repeat):
            x_rand = emerge()
            Debugger().debug_sampling(x_rand, self.check_poly, switch=self.debug)
            if is_free(x_rand):
                if not exist(x_rand):
                    return self.StateNode(tuple(x_rand))
        return self.StateNode(tuple(x_rand))

    def nearest(self, x_rand):  # type: (StateNode) -> StateNode
        """find the state in the tree which is nearest to the sampled state.
        And fill the g, hl and fl properties of the sampled state.
        """

        def replenish(x_n, x_r):
            x_r.g, x_r.hl = x_n.g + self.cost(x_n, x_r), self.cost(x_r, self.gain)
            x_r.fl = x_r.g + x_r.hl

        # quick shot
        if self.collision_free(self.root, x_rand):
            x_nearest = self.root
        else:
            costs = list(map(lambda x: self.cost(x, x_rand), self.vertices))
            x_nearest = self.vertices[int(np.argmin(costs))]
        replenish(x_nearest, x_rand)
        Debugger().debug_nearest_searching(x_nearest.state, switch=self.debug)
        return x_nearest

    def least(self, x_rand):  # type: (StateNode) -> StateNode
        def replenish(x_n, x_r):
            x_r.g, x_r.hl = x_n.g + self.cost(x_n, x_r), self.cost(x_r, self.gain)
            x_r.fl = x_r.g + x_r.hl

        # quick shot
        if self.collision_free(self.root, x_rand):
            x_least = self.root
        else:
            nodes = filter(lambda x: self.collision_free(x, x_rand), self.vertices)
            if nodes:
                costs = list(map(lambda x: x.g + self.cost(x, x_rand), nodes))
                x_least = nodes[int(np.argmin(costs))]
            else:
                x_least = None
        if x_least:
            replenish(x_least, x_rand)
            Debugger().debug_nearest_searching(x_least.state, switch=self.debug)
        return x_least

    def benefit(self, x_new):
        words = 'Benefit: {}/ ({}, {})'.format(x_new.fl <= self.x_best.fu, x_new.fl, self.x_best.fu)
        Debugger.breaker(words, switch=self.debug)
        return x_new.fl < self.x_best.fu

    def collision_free(self, x_from, x_to):  # type: (StateNode, StateNode) -> bool
        """check if the path from one state to another state collides with any obstacles or not."""
        # making contours of the curve
        states = reeds_shepp.path_sample(x_from.state, x_to.state, 1. / self.maximum_curvature, 0.3)
        # states.append(tuple(x_to.state))  # include the end point
        contours = [self.transform(self.check_poly, s[0], s[1], s[2]) for s in states]
        contours = [np.floor(con / self.grid_res + self.grid_map.shape[0] / 2.).astype(int) for con in contours]
        # making mask
        mask = np.zeros_like(self.grid_map, dtype=np.uint8)
        [cv2.fillPoly(mask, [con], 255) for con in contours]
        # checking
        result = np.bitwise_and(mask, self.grid_map)
        Debugger().debug_collision_checking(states, self.check_poly, np.all(result < self.obstacle), switch=self.debug)
        return np.all(result < self.obstacle)

    @staticmethod
    @numba.njit
    def transform(poly, x, y, a):
        pts = poly.transpose()
        xyo = np.array([[x], [y]])
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return (np.dot(rot, pts) + xyo).transpose()

    def attach(self, x_nearest, x_new):  # type: (StateNode, StateNode) -> None
        """add the new state to the tree and complement other values.
        And fill the hu and fu properties of x_new.
        """
        x_new.match(x_nearest)
        available = self.collision_free(x_new, self.gain)
        x_new.hu = x_new.hl if available else np.inf
        x_new.fu = x_new.g + x_new.hu
        x_new.status = 0 if available else 1
        self.vertices.append(x_new)
        Debugger().debug_attaching(x_nearest, x_new, 1. / self.maximum_curvature, switch=self.debug)

    def rewire(self, x_new, gamma=0.2):  # type: (StateNode, float) -> None
        """rewiring tree by the new state."""

        def recheck(x):
            available, cost = self.collision_free(x_new, x), self.cost(x_new, x)
            if available and x.g > x_new.g + cost:
                Debugger().debug_rewiring(x, x_new.g + cost, switch=self.debug)
                x.g = x_new.g + cost
                x.fu, x.fl = x.g + x.hu, x.g + x.hl
                x.rematch(x_new)

        xs = filter(lambda x: x.g > x_new.g + gamma, self.vertices)
        Debugger().debug_rewiring_check(xs, x_new, switch=self.debug)
        map(recheck, xs)

    def cost(self, x_from, x_to):  # type: (StateNode, StateNode) -> float
        """calculate the cost from one state to another state"""
        return reeds_shepp.path_length(x_from.state, x_to.state, 1. / self.maximum_curvature)

    def best(self):
        return sorted(self.vertices, key=lambda x: x.fu)[0]

    def spare(self, space=None):
        return sorted(self.vertices if not space else space, key=lambda x: x.hl)[0]

    def path(self):  # type: () -> List[RRTStar.StateNode]
        """extract the planning result. including the goal state if the the goal is available"""
        x_best = self.best()
        if x_best.fu < np.inf:
            p = x_best.trace()
            self.gain.g = self.gain.fu = x_best.fu
            p.append(self.gain)
            return p
        else:
            x_spare = self.spare()
            return x_spare.trace()

    def trajectory(self, a_cc=3, v_max=20, res=0.5):
        """
        planning velocity for a path to generate a trajectory.
        """

        def interpolate(q_ori, segment_type, length, radius):
            x0, y0, a0 = q_ori[0], q_ori[1], q_ori[2]
            transfer = np.array([[np.cos(a0), -np.sin(a0), 0., x0], [np.sin(a0), np.cos(a0), 0., y0], [0., 0., 1., a0]])
            sign, phi = np.sign(length), np.fabs(length) / radius
            if segment_type == 1:
                r = 2 * radius * np.sin(phi / 2.)
                x_lcs = np.array([[r * sign * np.cos(phi / 2.)], [r * np.sin(phi / 2.)], [sign * phi], [1]])
            elif segment_type == 3:
                r = 2 * radius * np.sin(phi / 2.)
                x_lcs = np.array([[r * sign * np.cos(phi / 2.)], [- r * np.sin(phi / 2.)], [- sign * phi], [1]])
            else:
                x_lcs = np.array([[length], [0], [0], [1]])
            x_tar = np.dot(transfer, x_lcs)
            return x_tar[0, 0], x_tar[1, 0], x_tar[2, 0]

        def extract_segments(q_from, q_to):
            segments.extend(reeds_shepp.path_type(q_from, q_to, 1. / self.maximum_curvature))
            return q_to

        def extract_discontinuities(q0, sgs):
            sg0, sg1 = sgs[0], sgs[1]
            q1 = interpolate(q0, sg0[0], sg0[1], 1. / self.maximum_curvature)
            if sg0[1] * sg1[1] < 0:
                discontinuities.append(self.Configuration(q1, v=0))
            return q1

        def plan_motions(sector):
            q0, v0, q1, v1 = sector[0].state, sector[0].v, sector[1].state, sector[1].v
            extent = reeds_shepp.path_length(q0, q1, 1. / self.maximum_curvature)
            acc = min([(v_max ** 2 - v1 ** 2) / extent, a_cc])
            vcc = np.sqrt(v1 ** 2 + acc * extent)
            samples = reeds_shepp.path_sample(q0, q1, 1. / self.maximum_curvature, res)
            for i, sample in enumerate(samples):
                if i * res < extent / 2.:
                    vt = min([np.sqrt(v0 ** 2 + 2 * acc * (i * res)), vcc])
                else:
                    vt = min([np.sqrt(v1 ** 2 + 2 * acc * (extent - i * res)), vcc])
                motions.append(self.Configuration(sample[:3], k=sample[3], v=np.sign(sample[4]) * vt))

        segments = []  # type: List[(float, float)]
        path = [tuple(node.state) for node in self.path()]
        reduce(extract_segments, path)
        segments = zip(segments[:-1], segments[1:])  # type: List[(Tuple[float], Tuple[float])]

        discontinuities = []  # type: List[(Tuple[float], float)]
        segments.insert(0, tuple(self.root.state))
        reduce(extract_discontinuities, segments)
        discontinuities.append(self.Configuration().from_state_node(self.gain))
        discontinuities.insert(0, discontinuities.append(self.Configuration().from_state_node(self.root)))

        motions = []
        sectors = zip(discontinuities[:-1], discontinuities[1:])
        map(plan_motions, sectors)
        motions.append(self.Configuration().from_state_node(self.gain))
        return motions

    class Configuration(object):
        def __init__(self, state=(), v=None, k=None):
            self.state = np.array(state)
            self.v, self.k = v, k

        def from_state_node(self, state_node):
            # type: (RRTStar.StateNode) -> RRTStar.Configuration
            self.state = state_node.state
            self.v, self.k = state_node.v, state_node.k
            return self

    class StateNode(object):
        def __init__(self, state=()):
            # type: (tuple) -> None
            self.state = np.array(state)  # state of the Node, a tuple (x, y, orientation)
            self.g = np.inf  # cost from root to here.
            self.hu = np.inf  # cost from here to goal if available.
            self.hl = np.inf  # cost from here to goal if not available.
            self.fu = self.g + self.hu
            self.fl = self.g + self.hl
            self.parent = None  # type: Optional[RRTStar.StateNode]
            self.children = []  # type: List[RRTStar.StateNode]
            self.status = 0  # 0 for safe, 1 for dangerous.
            self.v, self.k = 0, None  # velocity of the state, curvature of the state (related to the steering angle)

        def match(self, x_parent):
            # type: (RRTStar.StateNode) -> None
            """
            add a state as parent.
            """
            self.parent = x_parent
            x_parent.children.append(self)

        def remove(self):
            if self.parent:
                self.parent.children.remove(self)
                self.parent = None

        def rematch(self, x_new_parent):
            if self.parent:
                self.parent.children.remove(self)
                self.match(x_new_parent)

        def trace(self):  # type: ()->List[RRTStar.StateNode]
            p, ptr = [self], self.parent
            while ptr:
                p.append(ptr)
                ptr = ptr.parent
            p.reverse()
            return p

        def lcs2gcs(self, origin):
            # type: (RRTStar.StateNode) -> RRTStar.StateNode
            """
            transform self's coordinate from local coordinate system (LCS) to global coordinate system (GCS)
            :param origin: the tuple the coordinate (in GCS) of the origin of LCS.
            """
            xo, yo, ao = origin[0], origin[1], origin[2]
            x = self.state[0] * np.cos(ao) - self.state[1] * np.sin(ao) + xo
            y = self.state[0] * np.sin(ao) + self.state[1] * np.cos(ao) + yo
            a = self.state[2] + ao
            self.state = np.array((x, y, a))
            return self

        def gcs2lcs(self, origin):
            # type: (RRTStar.StateNode) -> RRTStar.StateNode
            """
            transform self's coordinate from global coordinate system (LCS) to local coordinate system (GCS)
            :param origin: the circle-node contains the coordinate (in GCS) of the origin of LCS.
            """
            xo, yo, ao = origin[0], origin[1], origin[2]
            x = (self.state[0] - xo) * np.cos(ao) + (self.state[1] - yo) * np.sin(ao)
            y = -(self.state[0] - xo) * np.sin(ao) + (self.state[1] - yo) * np.cos(ao)
            a = self.state[2] - ao
            self.state = np.array((x, y, a))
            return self


class BiRRTStar(RRTStar):
    def __init__(self):
        super(BiRRTStar, self).__init__()
        self.s_vertices = None
        self.g_vertices = None

    def preset(self, start, goal, grid_map, grid_res, grid_ori, obstacle, heuristic):
        self.grid_map, self.grid_res, self.grid_ori, self.obstacle = grid_map, grid_res, grid_ori, obstacle
        self.heuristic = heuristic
        self.start, self.goal = start, goal

        self.start.g, self.start.hl = 0, self.cost(start, goal)
        self.start.hu = self.start.hl if self.collision_free(start, goal) else np.inf
        self.start.fl, self.start.fu = self.start.g + self.start.hl, self.start.g + self.start.hu

        self.goal.g, self.goal.hl = 0, self.cost(goal, start)
        self.goal.hu = self.goal.hl if self.collision_free(goal, start) else np.inf
        self.goal.fl, self.goal.fu = self.goal.g + self.goal.hl, self.goal.g + self.goal.hu

        self.root, self.gain = self.start, self.goal
        self.s_vertices = [self.start]
        self.g_vertices = [self.goal]
        self.x_best = self.root
        return self

    def swap(self, i):
        self.branch_and_bound(self.g_vertices)
        self.branch_and_bound(self.s_vertices)
        if self.root is self.start:
            self.root = self.goal
            self.gain = self.start
            self.vertices = self.g_vertices
            n = -((i/2) % len(self.heuristic)) - 1 if self.heuristic else i
            Debugger.breaker('swap: goal -> start, {}, {}'.format(i, n), self.debug)
            return n
        else:
            self.root = self.start
            self.gain = self.goal
            self.vertices = self.s_vertices
            n = (i/2) % len(self.heuristic) if self.heuristic else i
            Debugger.breaker('swap: start -> goal, {}, {}'.format(i, n), self.debug)
            return n

    def planning(self, times, repeat=10, optimize=False, debug=False):
        """main flow."""
        self.debug = debug
        past = time.time()
        for i in range(times):
            n = self.swap(i)
            x_new = self.sample_free(n, repeat)
            x_nearest = self.nearest(x_new) if not optimize else self.least(x_new)
            if x_nearest and self.benefit(x_new) and self.collision_free(x_nearest, x_new):
                self.attach(x_nearest, x_new)
                self.rewire(x_new)
                self.x_best = self.best_of_all()
                self.connect_graphs(x_new)
            Debugger().debug_planned_path(self, i, switch=self.debug)
            Debugger().debug_planning_hist(self, i, (time.time() - past), switch=True)

    def best_of_all(self):
        x_new_best = self.best()
        return x_new_best if x_new_best.fu < self.x_best.fu else self.x_best

    def connect_graphs(self, x_new):
        if self.root is self.start:
            vs = self.g_vertices
        else:
            vs = self.s_vertices
        costs = map(lambda x: self.cost(x_new, x), vs)
        x_nearest, cost = vs[int(np.argmin(costs))], np.min(costs)
        Debugger().debug_connect_graphs(x_nearest.state, x_new.g+cost+x_nearest.g, self.x_best.fu, switch=self.debug)
        if x_new.g + cost + x_nearest.g < self.x_best.fu:
            if self.collision_free(x_new, x_nearest):
                x_nearest.fu = x_new.fu = x_new.g + cost + x_nearest.g
                x_new.hu = x_new.fu - x_new.g
                x_nearest.hu = x_nearest.fu - x_nearest.g
                x_new.neighbor = x_nearest
                x_nearest.neighbor = x_new
                self.x_best = x_new

    def path(self):
        x_best = self.x_best
        if x_best.fu < np.inf:
            p = x_best.trace()
            if hasattr(x_best, 'neighbor') and x_best.neighbor:
                p1 = x_best.neighbor.trace()
                p1.reverse()
                p.extend(p1)
                if p[-1] is self.start:
                    p.reverse()
                return p
            else:
                if p[0] is self.start:
                    p.append(self.gain)
                else:
                    p.append(self.start)
                    p.reverse()
                return p
        else:
            x_spare = self.spare(space=self.s_vertices)
            return x_spare.trace()
