#!/usr/bin/env python
"""
module to define base data type
"""
from copy import deepcopy
from enum import Enum
from typing import List  # pylint: disable=unused-import
from anytree import NodeMixin
from recordclass import recordclass
import geometry_msgs.msg as gmm
import numpy as np
import pyquaternion
import tf.transformations as tfm


class Vector3D(object):
    """
    Vector 3D helper class.
    """
    __slots__ = ('x', 'y', 'z', 'vec')

    def __init__(self, vec=(0.0, 0.0, 0.0)):
        """
        :param vec: 1-D array, in order: (x, y, z)
        """
        self.x = vec[0]
        self.y = vec[1]
        self.z = vec[2]
        self.vec = np.array(vec, dtype=np.float)

    def __add__(self, other):
        """
        :type other: Vector3D or SubClass
        """
        return type(other)(vec=self.vec + other.vec)

    def __mul__(self, other):
        """
        :type other: Vector3D or SubClass
        """
        return type(other)(vec=self.vec * other.vec)


class Rotation(object):
    """
    Class that represents a 3D rotation.
    All rotation angles are stored in radians.
    """
    __slots__ = ('r', 'p', 'y', 'q')

    def __init__(self, rpy=(0.0, 0.0, 0.0)):
        """
        :param rpy: 1-D array, in order: (roll, pitch, yaw) in radians
        """
        self.r = rpy[0]
        self.p = rpy[1]
        self.y = rpy[2]
        self.q = self._get_quaternion()

    def __add__(self, other):
        # type: (Rotation) -> Rotation
        return Rotation(
            rpy=(self.r + other.r, self.p + other.p, self.y + other.y))

    def __mul__(self, other):
        # type: (Rotation) -> Rotation
        q = self.q * other.q
        return Rotation(rpy=tfm.euler_from_quaternion([q.x, q.y, q.z, q.w]))

    def _get_quaternion(self):
        """
        :return:
        """
        qua = tfm.quaternion_from_euler(self.r, self.p, self.y)
        return pyquaternion.Quaternion(w=qua[3], x=qua[0], y=qua[1], z=qua[2])

    def rotation_matrix(self, inv=False):
        """
        :param inv:
        :return:
        """
        return (self.q.conjugate.rotation_matrix if inv
                else self.q.rotation_matrix)

    def transformation(self, inv=False):
        """
        :param inv:
        :return:
        """
        return (self.q.conjugate.transformation_matrix if inv
                else self.q.transformation_matrix)

    def to_ros(self):
        # type: () -> gmm.Quaternion
        """
        convert Posture to Quaternion of geometry_msgs
        :return: ros geometry_msgs/Quaternion
        """
        return gmm.Quaternion(w=self.q.w, x=self.q.x, y=self.q.y, z=self.q.z)

    @property
    def info(self):
        # type: () -> str
        """
        :return: string of info of this posture
        """
        return "roll: {}, pitch: {}, yaw: {}".format(
            np.degrees(self.r), np.degrees(self.p), np.degrees(self.y))


class Location(Vector3D):
    """
    Represents a location in the world (in meters).
    """

    def __init__(self, vec=(0.0, 0.0, 0.0)):
        """
        :param vec: 1-D array, in order: (x, y, z)
        """
        super(Location, self).__init__(vec)

    def distance(self, other):
        # type: (Location) -> float
        """
        Computes the Euclidean distance in meters from this point to the other.
        """
        return np.linalg.norm(self.vec - other.vec)

    def transformation(self, inv=False):
        """
        :param inv:
        :return:
        """
        t_matrix = np.identity(self.vec.size + 1)
        t_matrix[:-1, -1] = -self.vec if inv else self.vec
        return t_matrix

    def to_ros(self):
        # type: () -> gmm.Point
        """
        convert Point3D to Point of geometry_msgs
        :return: geometry_msgs/Point
        """
        return gmm.Point(x=self.x, y=self.y, z=self.z)

    @property
    def info(self):
        # type: () -> str
        """
        :return: string of info of this point
        """
        return "x: {}, y: {}, z: {}".format(self.x, self.y, self.z)


class Velocity(Vector3D):
    """
    Represents the velocity of a vehicle (in m/s).
    """

    def __init__(self, vec=(0.0, 0.0, 0.0), speed=None):
        """
        :param vec: speed along x, y, z, in order [x, y, z]
        """
        super(Velocity, self).__init__(vec)
        self.speed = np.linalg.norm(self.vec) if not speed else speed

    def reset(self, vec):
        """
        :param vec:
        :return:
        """
        self.x = vec[0]
        self.y = vec[1]
        self.z = vec[2]
        self.vec = np.array(vec, dtype=np.float)
        self.speed = np.linalg.norm(self.vec)

    def transformation(self, inv=False):
        """
        :param inv:
        :return:
        """
        t_matrix = np.identity(self.vec.size + 1)
        t_matrix[:-1, -1] = -self.vec if inv else self.vec
        return t_matrix

    @property
    def info(self):
        # type: () -> str
        """
        :return: string of info about velocity
        """
        return "speed: {}, x: {}, y:{}, z: {}".format(self.speed, self.x,
                                                      self.y, self.z)


class GeoReference(Enum):
    """
    Represents the Geo reference for transformation between xy and lat,lon.
    """
    LATITUDE = 4.9000000000000000e+1
    LONGITUDE = 8.0000000000000000e+0


class NStatus(Enum):
    """
    Represents the Status of Node: RoadNode and TreeNode
    """
    FREE = 0  # type: int
    STOP = 2  # type: int


class WStatus(Enum):
    """
    Represents the Status of the Planned Way.
    Push means acceleration, Pull means deceleration.
    """
    PUSH = 0  # type: int
    PULL = 2  # type: int


class C2GoType(Enum):
    """
    Represents the Status of the curve from a TreeNode to another one.
    Free means collision, Unfree means no collision.
    """
    CLOSE = 0  # type: int
    BREAK = 1  # type: int


class PlannerStatus(Enum):
    """
    Represents the Status of the Planner.
    """
    STANDBY = -1  # type: int
    RUNNING = 0  # type: int
    ARRIVED = 1  # type: int


class RegionType(Enum):
    """
    Represents the type of the regions.
    """
    AVAILABLE = 1
    OCCUPIED = 255
    UNKNOWN = 127


class State(object):
    """
    Represents a state or said a configuration.
    """
    __slots__ = ('location', 'rotation', 'velocity')

    def __init__(self, location, rotation, velocity):
        # type: (Location, Rotation, Velocity) -> None
        self.location = location
        self.rotation = rotation
        self.velocity = velocity

    def transform(self, state, inv=True):
        # type: (State, bool) -> State
        """
        if inv is True, transform state to self.state's frame
        if inv is False, transform state in self.state's frame to global frame.
        :param inv:
        :param state: State class
        :return: state in the local-frame of self.state
        """
        velocity = Velocity(np.dot(self.velocity.transformation(inv=inv),
                                   np.append(state.velocity.vec, 1))[:-1])
        location = Location(np.dot(self.transformation(inv=inv),
                                   np.append(state.location.vec, 1))[:-1])
        rotation = Rotation(tfm.euler_from_matrix(
            np.dot(state.rotation.transformation(),
                   self.rotation.transformation(inv=inv))))
        return State(location=location, rotation=rotation, velocity=velocity)

    def transformation(self, inv=False):
        """
        :param inv:
        :return:
        """
        return (np.dot(self.rotation.transformation(inv=inv),
                       self.location.transformation(inv=inv)) if inv
                else np.dot(self.location.transformation(inv=inv),
                            self.rotation.transformation(inv=inv)))

    def to_ros_pose(self):
        # type: () -> gmm.Pose
        """
        convert State to Pose of ros geometry_msgs
        """
        return gmm.Pose(position=self.location.to_ros(),
                        orientation=self.rotation.to_ros())

    @property
    def info(self):
        # type: () -> str
        """
        :return: string of info about State
        """
        return "location:\n\t{}\nrotation:\n\t{}\nvelocity:\n\t{}>".format(
            self.location.info, self.rotation.info, self.velocity.info)


class C2Go(object):
    """
    Represents the cost from a state to another one.
    """
    __slots__ = ('vec', 'scales', 'c2gtype', 'grade', 'reverse')

    def __init__(self, c2gtype=None, scales=(1.0, 0.0, 0.0, 0.),
                 vec=(np.inf, np.inf, np.inf, np.inf), grade=np.inf):
        """
        :param vec: 1-D array, in order [leg, k, dk, euler]
        """
        self.grade = grade
        self.reverse = False
        self.c2gtype = c2gtype
        self.vec = np.array(vec, dtype=np.float)
        self.scales = np.array(scales, dtype=np.float)

    def reset_vec(self, vec):
        """
        :param vec:
        :return:
        """
        self.vec = np.array(vec, dtype=np.float)

    def cost(self):
        # type: () -> float
        """
        :return: cost after mask with the scales
        """
        return float(np.dot(self.vec, self.scales))

    @property
    def info(self):
        # type: () -> str
        """
        :return: string of info about velocity
        """
        return "leg: {}, k: {}, dk: {}, euler:{}, c2gtype: {}".format(
            self.vec[0], self.vec[1], self.vec[2], self.vec[3], self.c2gtype)


RoadNode = recordclass(  # pylint: disable=invalid-name
    'RoadNode',
    ['state', 'status', 'cost2go'])


class TreeNode(NodeMixin):
    """
    Represents a node in tree for planning.
    """
    __slots__ = ('name', 'state', 'c2go', 'c2get', 'status')

    def __init__(self, name, state, c2go, c2get,  # pylint: disable=too-many-arguments
                 status, parent=None, children=None):
        # type: (str, State, C2Go, C2Go, NStatus.FREE, TreeNode, List[TreeNode]) -> None
        super(TreeNode, self).__init__()
        self.name = name
        self.state = state
        self.status = status
        self.parent = parent
        self.c2go = c2go
        self.c2get = c2get
        if children:
            self.children = children

    def to_road_node(self):
        # type: () -> RoadNode
        """
        convert TreeNode to RoadNode
        """
        return RoadNode(self.state, self.status, self.c2go)

    @property
    def replica(self):
        """
        a deepcopy of self without the parent and children
        :return:
        """
        return TreeNode(name=self.name,
                        state=deepcopy(self.state),
                        c2go=deepcopy(self.c2go),
                        c2get=deepcopy(self.c2get),
                        status=deepcopy(self.status))

    @property
    def info(self):
        # type: () -> str
        """
        :return: string of info about TreeNode
        """
        return ("name:\n\t{}\nstatus:\n\t{}\n"
                "state:\n\t{}\ncost2go:\n\t{}\ncost2get{}".format(
                    self.name, self.status, self.state.info,
                    self.c2go.info, self.c2get.info))
