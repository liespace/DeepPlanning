from copy import deepcopy
from typing import List
import numba
from numba import njit
import numpy as np
import reeds_shepp
import matplotlib.pyplot as plt


class OrientationSpaceExplorer(object):
    def __init__(self,
                 minimum_radius=0.2,
                 maximum_radius=2.96,  # 2.5
                 minimum_clearance=1.02,
                 neighbors=32,
                 maximum_curvature=0.2,
                 timeout=1.0,
                 overlap_rate=0.5):
        self.minimum_radius = minimum_radius
        self.maximum_radius = maximum_radius
        self.minimum_clearance = minimum_clearance
        self.neighbors = neighbors
        self.maximum_curvature = maximum_curvature
        self.timeout = timeout
        self.overlap_rate = overlap_rate
        # planning related
        self.start = None
        self.goal = None
        self.grid_ori = None
        self.grid_map = None
        self.grid_res = None
        self.grid_pad = None
        self.obstacle = 255

    def exploring(self, plotter=None):
        close_set, open_set = numba.typed.List(), [(self.start.f, self.start)]
        close_set.append((0., 0., 0., 0.)), close_set.pop()
        while open_set:
            circle = self.pop_top(open_set)
            if self.goal.f < circle.f:
                return True
            if not self.exist(circle, close_set):
                expansion = self.expand(circle)
                self.merge(expansion, open_set)
                if self.overlap(circle, self.goal) and circle.f < self.goal.g:
                    self.goal.g = circle.f
                    self.goal.f = self.goal.g + self.goal.h
                    self.goal.set_parent(circle)
                close_set.append((circle.x, circle.y, circle.a, circle.r))
            if plotter:
                plotter([circle])
        return False

    @property
    def circle_path(self):
        if self.goal:
            path, parent = [self.goal], self.goal.parent
            while parent:
                path.append(parent)
                parent = parent.parent
            path.reverse()
            return path
        return []

    def path(self):
        circles = self.circle_path
        if circles:
            return [(p.x, p.y, p.a, p.r) for p in circles]
        return []

    def initialize(self, start, goal, grid_map, grid_res, grid_ori, obstacle=255):
        # type: (CircleNode, CircleNode, np.ndarray, float, CircleNode, int) -> OrientationSpaceExplorer
        """
        :param start: start circle-node
        :param goal: goal circle-node
        :param grid_map: occupancy map(0-1), 2d-square, with a certain resolution: gird_res
        :param grid_res: resolution of occupancy may (m/pixel)
        :param obstacle: value of the pixels of obstacles region on occupancy map.
        """
        self.start, self.goal = start, goal
        self.grid_map, self.grid_res, self.grid_ori, self.obstacle = grid_map, grid_res, grid_ori, obstacle
        # padding grid map for clearance calculation
        s = int(np.ceil((self.maximum_radius + self.minimum_clearance) / self.grid_res))
        self.grid_pad = np.pad(self.grid_map, ((s, s), (s, s)), 'constant',
                               constant_values=((self.obstacle, self.obstacle), (self.obstacle, self.obstacle)))
        # complete the start and goal
        self.start.r, self.start.g = self.clearance(self.start) - self.minimum_clearance, 0
        self.start.h = reeds_shepp.path_length(
            (start.x, start.y, start.a), (self.goal.x, self.goal.y, self.goal.a), 1. / self.maximum_curvature)
        self.goal.r, self.goal.h, self.goal.g = self.clearance(self.goal) - self.minimum_clearance, 0, np.inf
        self.start.f, self.goal.f = self.start.g + self.start.h, self.goal.g + self.goal.h
        return self

    @staticmethod
    def merge(expansion, open_set):
        """
        :param expansion: expansion is a set in which items are unordered.
        :param open_set: we define the open set as a set in which items are sorted from Large to Small by cost.
        """
        open_set.extend(zip(map(lambda x: x.f, expansion), expansion))
        open_set.sort(reverse=True)

    @staticmethod
    def pop_top(open_set):
        """
        :param open_set: we define the open set as a set in which items are sorted from Large to Small by cost.
        """
        return open_set.pop()[-1]

    def exist(self, circle, close_set):
        state = (circle.x, circle.y, circle.a)
        return self.jit_exist(state, close_set, self.maximum_curvature)

    @staticmethod
    @njit
    def jit_exist(state, close_set, maximum_curvature):
        def distance(one, another, curvature):
            euler = np.sqrt((one[0] - another[0]) ** 2 + (one[1] - another[1]) ** 2)
            angle = np.abs(one[2] - another[2])
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            # angle = np.pi - angle if angle > np.pi / 2 else angle
            heuristic = angle / curvature
            return euler if euler > heuristic else heuristic
        for item in close_set:
            if distance(state, item, maximum_curvature) < item[-1] - 0.1:
                return True
        return False

    def overlap(self, circle, goal):
        """
        check if two circles overlap with each other
        in a certain margin (overlap_rate[e.g., 50%] of the radius of the smaller circle),
        which guarantees enough space for a transition motion.
        """
        return self.jit_overlap((circle.x, circle.y, circle.r), (goal.x, goal.y, goal.r), self.overlap_rate)

    @staticmethod
    @njit
    def jit_overlap(circle, goal, rate):
        euler = np.sqrt((circle[0] - goal[0]) ** 2 + (circle[1] - goal[1]) ** 2)
        r1, r2 = min([circle[2], goal[2]]), max([circle[2], goal[2]])
        return euler < r1 * rate + r2

    def expand(self, circle):
        def buildup(state):
            child = self.CircleNode(state[0], state[1], state[2], state[3])
            # build the child
            child.set_parent(circle)
            # child.h = self.distance(child, self.goal)
            child.h = reeds_shepp.path_length(
                (child.x, child.y, child.a), (self.goal.x, self.goal.y, self.goal.a), 1. / self.maximum_curvature)
            child.g = circle.g + reeds_shepp.path_length(
                (circle.x, circle.y, circle.a), (child.x, child.y, child.a), 1. / self.maximum_curvature)
            child.f = child.g + child.h
            # add the child to expansion set
            return child

        neighbors = self.jit_neighbors((circle.x, circle.y, circle.a), circle.r, self.neighbors)
        children = self.jit_children(
            neighbors, (self.grid_ori.x, self.grid_ori.y, self.grid_ori.a), self.grid_pad, self.grid_map, self.grid_res,
            self.maximum_radius, self.minimum_radius, self.minimum_clearance, self.obstacle)
        return map(buildup, children)

    @staticmethod
    @njit
    def jit_children(neighbors, origin, grid_pad, grid_map, grid_res, maximum_radius, minimum_radius, minimum_clearance, obstacle):
        def clearance(state):
            s_x, s_y, s_a = origin[0], origin[1], origin[2]
            c_x, c_y, c_a = state[0], state[1], state[2]
            x = (c_x - s_x) * np.cos(s_a) + (c_y - s_y) * np.sin(s_a)
            y = -(c_x - s_x) * np.sin(s_a) + (c_y - s_y) * np.cos(s_a)
            u = int(np.floor(y / grid_res + grid_map.shape[0] / 2))
            v = int(np.floor(x / grid_res + grid_map.shape[0] / 2))
            size = int(np.ceil((maximum_radius + minimum_clearance) / grid_res))
            subspace = grid_pad[u:u + 2 * size + 1, v:v + 2 * size + 1]
            rows, cols = np.where(subspace >= obstacle)
            if len(rows):
                row, col = np.fabs(rows - size) - 1, np.fabs(cols - size) - 1
                rs = np.sqrt(row ** 2 + col ** 2) * grid_res
                return rs.min()
            else:
                return size * grid_res
        children = numba.typed.List()
        children.append((0., 0., 0., 0.)), children.pop()
        for neighbor in neighbors:
            r = min([clearance(neighbor) - minimum_clearance, maximum_radius])
            if r > minimum_radius:
                children.append((neighbor[0], neighbor[1], neighbor[2], r))
        return children

    @staticmethod
    @njit
    def jit_neighbors(state, radius, number):
        def lcs2gcs(point):
            x, y, a = point
            xo, yo, ao = state
            x1 = x * np.cos(ao) - y * np.sin(ao) + xo
            y1 = x * np.sin(ao) + y * np.cos(ao) + yo
            a1 = a + ao
            return x1, y1, a1
        neighbors = numba.typed.List()
        neighbors.append((0., 0., 0.)), neighbors.pop()
        for n in np.radians(np.linspace(-90, 90, number / 2)):
            neighbor = (radius * np.cos(n), radius * np.sin(n), n)
            opposite = (radius * np.cos(n + np.pi), radius * np.sin(n + np.pi), n)
            neighbor = lcs2gcs(neighbor)
            opposite = lcs2gcs(opposite)
            neighbors.extend([neighbor, opposite])
        return neighbors

    def clearance(self, circle):
        origin, coord = (self.grid_ori.x, self.grid_ori.y, self.grid_ori.a), (circle.x, circle.y, circle.a)
        return self.jit_clearance(coord, origin, self.grid_pad, self.grid_map, self.grid_res,
                                  self.maximum_radius, self.minimum_clearance, self.obstacle)

    @staticmethod
    @njit
    def jit_clearance(coord, origin, grid_pad, grid_map, grid_res, maximum_radius, minimum_clearance, obstacle):
        s_x, s_y, s_a = origin[0], origin[1], origin[2]
        c_x, c_y, c_a = coord[0], coord[1], coord[2]
        x = (c_x - s_x) * np.cos(s_a) + (c_y - s_y) * np.sin(s_a)
        y = -(c_x - s_x) * np.sin(s_a) + (c_y - s_y) * np.cos(s_a)
        u = int(np.floor(y / grid_res + grid_map.shape[0] / 2))
        v = int(np.floor(x / grid_res + grid_map.shape[0] / 2))
        size = int(np.ceil((maximum_radius + minimum_clearance) / grid_res))
        subspace = grid_pad[u:u + 2 * size + 1, v:v + 2 * size + 1]
        rows, cols = np.where(subspace >= obstacle)
        if len(rows):
            row, col = np.fabs(rows - size) - 1, np.fabs(cols - size) - 1
            rs = np.sqrt(row ** 2 + col ** 2) * grid_res
            return rs.min()
        else:
            return size * grid_res

    def plot_circles(self, circles):
        # type: (List[CircleNode]) -> None
        for circle in circles:
            c = deepcopy(circle).gcs2lcs(self.grid_ori)
            cir = plt.Circle(xy=(c.x, c.y), radius=c.r, color=(0.5, 0.8, 0.5), alpha=0.6)
            arr = plt.arrow(x=c.x, y=c.y, dx=0.5 * np.cos(c.a), dy=0.5 * np.sin(c.a), width=0.1)
            plt.gca().add_patch(cir)
            plt.gca().add_patch(arr)

    @staticmethod
    def plot_grid(grid_map, grid_res):
        # type: (np.ndarray, float) -> None
        """plot grid map"""
        row, col = grid_map.shape[0], grid_map.shape[1]
        indexes = np.argwhere(grid_map == 255)
        xy2uv = np.array([[0., 1. / grid_res, row / 2.], [1. / grid_res, 0., col / 2.], [0., 0., 1.]])
        for index in indexes:
            uv = np.array([index[0], index[1], 1])
            xy = np.dot(np.linalg.inv(xy2uv), uv)
            rect = plt.Rectangle((xy[0] - grid_res, xy[1] - grid_res), grid_res, grid_res, color=(1.0, 0.1, 0.1))
            plt.gca().add_patch(rect)

    circle_node = numba.deferred_type()
    spec = [
        ('x', numba.float64),
        ('y', numba.float64),
        ('a', numba.float64),
        ('r', numba.optional(numba.float64)),
        ('h', numba.float64),
        ('g', numba.float64),
        ('f', numba.float64),
        ("parent", numba.optional(circle_node)),
        ('children', numba.optional(numba.types.List(circle_node)))]

    # @numba.jitclass(spec)
    class CircleNode(object):

        def __init__(self, x=None, y=None, a=None, r=None):
            self.x = x
            self.y = y
            self.a = a
            self.r = r
            self.h = np.inf  # cost from here to goal, heuristic distance or actual one
            self.g = np.inf  # cost from start to here, actual distance
            self.f = self.h + self.g
            self.parent = None
            self.children = None

        def set_parent(self, circle):
            self.parent = circle

        def lcs2gcs(self, circle):
            # type: (OrientationSpaceExplorer.CircleNode) -> OrientationSpaceExplorer.CircleNode
            """
            transform self's coordinate from local coordinate system (LCS) to global coordinate system (GCS)
            :param circle: the circle-node contains the coordinate (in GCS) of the origin of LCS.
            """
            xo, yo, ao = circle.x, circle.y, circle.a
            x = self.x * np.cos(ao) - self.y * np.sin(ao) + xo
            y = self.x * np.sin(ao) + self.y * np.cos(ao) + yo
            a = self.a + ao
            self.x, self.y, self.a = x, y, a
            return self

        def gcs2lcs(self, circle):
            # type: (OrientationSpaceExplorer.CircleNode) -> OrientationSpaceExplorer.CircleNode
            """
            transform self's coordinate from global coordinate system (LCS) to local coordinate system (GCS)
            :param circle: the circle-node contains the coordinate (in GCS) of the origin of LCS.
            """
            xo, yo, ao = circle.x, circle.y, circle.a
            x = (self.x - xo) * np.cos(ao) + (self.y - yo) * np.sin(ao)
            y = -(self.x - xo) * np.sin(ao) + (self.y - yo) * np.cos(ao)
            a = self.a - ao
            self.x, self.y, self.a = x, y, a
            return self

    # define the deferred type
    # circle_node.define(CircleNode.class_type.instance_type)
