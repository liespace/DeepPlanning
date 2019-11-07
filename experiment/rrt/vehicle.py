#!/usr/bin/env python
"""
Vehicle helper module, to handle operation about vehicle.
"""
from collections import namedtuple
from recordclass import recordclass
import tf.transformations as tfm
import geometry_msgs.msg as gmm
import numpy as np
from dtype import State, Location, Rotation, Velocity

BoundingBox = recordclass(
    'BoundingBox',
    ['length', 'width', 'height', 'vbr', 'box2d'])
"""
# box length (m). # box width (m).
# box height (m).
# resolution of 2d grid bounding box (m/cell).
# 2d grid bounding box, set of 2D points.
"""

PhysAttrs = namedtuple(
    'PhysAttrs',
    ['mtr', 'loa', 'lod', 'laa', 'speed', 'weight'])
"""
# min turning radius (m)
# max longitudinal acceleration (N/kg)
# max longitudinal deceleration (N/kg)
# max lateral acceleration (N/kg)
# max speed (m/s)
# weight of the vehicle (kg)
"""


class VehicleInfo(object):
    """
    Represents the information of the vehicle
    """

    def __init__(self):
        self.bbox = BoundingBox(length=4.8, width=2.0, height=1.6, vbr=0.2,
                                box2d=None)
        self.attrs = PhysAttrs(mtr=5.0, loa=6.0, lod=4.0, laa=4.0,
                               speed=15, weight=2000.)
        self.bbox.box2d = self.bbox2d

    @property
    def bbox2d(self):
        # type: () -> np.ndarray
        """
        :return: a 2D-point set. (3, N) size matrix, each row denotes a point.
        """
        length, width, vbr = self.bbox.length, self.bbox.width, self.bbox.vbr
        wic, hic = int(length / vbr) + 1, int(width / vbr) + 1
        x = np.array(range(wic) * hic) * vbr - length / 2.
        y = np.repeat(np.array(range(hic)), wic) * vbr - width / 2.
        ones = np.ones(wic * hic)
        return np.stack((x, y, ones))

    def bbox2d_at(self, state=None):
        # type: (State) -> np.ndarray
        """
        generate the 2D grid bounding box with the state.
        :param state: state of vehicle box
        :return: bounding box with the state.
        """
        theta = state.rotation.y
        tf_matrix = np.array([[np.cos(theta), -np.sin(theta), state.location.x],
                              [np.sin(theta), np.cos(theta), state.location.y],
                              [0., 0., 1.]])
        return np.dot(tf_matrix, self.bbox.box2d)


class VehicleStatus(object):
    """
    Vehicle Status helper class.
    """
    __slots__ = ('pose', 'twist', 'accel', 'seq')

    def __init__(self, pose=gmm.Pose(), twist=gmm.Twist(), accel=gmm.Accel(),
                 seq=None):
        # type: (gmm.Pose, gmm.Twist, gmm.Accel, int) -> None
        self.pose = pose
        self.twist = twist
        self.accel = accel
        self.seq = seq

    def refresh(self, pose=None, twist=None, accel=None, seq=None):
        # type: (gmm.Pose, gmm.Twist, gmm.Accel, int) -> None
        """
        update vehicle status
        """
        self.pose = pose if pose else None
        self.twist = twist if twist else None
        self.accel = accel if accel else None
        self.seq = seq if seq else None

    @property
    def location(self):
        # type: () -> Location
        """
        the Location of the vehicle
        """
        pos = self.pose.position
        return Location(vec=(pos.x, pos.y, pos.z))

    @property
    def rotation(self):
        # type: () -> Rotation
        """
        the rotation or said orientation of vehicle
        """
        ori = self.pose.orientation
        rpy = tfm.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        return Rotation(rpy=rpy)

    @property
    def velocity(self):
        # type: () -> Velocity
        """
        Velocity of the vehicle
        :return: Velocity
        """
        vel = self.twist.linear
        return Velocity(vec=(vel.x, vel.y, vel.z))

    @property
    def state(self):
        # type: () -> State
        """
        State of the vehicle
        :return: State
        """
        return State(location=self.location,
                     rotation=self.rotation,
                     velocity=self.velocity)

    @property
    def info(self):
        """
        :return: string of info of vehicle status
        """
        return "Pose: {}, Twist: {}, Accel: {}, Seq: {}".format(
            self.pose, self.twist, self.accel, self.seq)


class Vehicle(object):
    """
    Vehicle helper class.
    """
    __slots__ = ('vinfo', 'status')

    def __init__(self, info=VehicleInfo(), status=VehicleStatus()):
        # type: (VehicleInfo, VehicleStatus) -> None
        self.vinfo = info
        self.status = status

    def is_turnable(self, k_s):
        # type: (list) -> bool
        """
        check if the turning radius is valid
        :param k_s: list of curvatures
        :return: True, if the turning radius >= minimum turning radius.
                 False, if the turning radius < minimum turning radius.
        """
        turn_radius = 1 / (np.fabs(k_s).max() + 1e-10)
        # rospy.logdebug('{} < {}(Mini.T.R)'.format(turn_radius, mtr))
        return turn_radius >= self.vinfo.attrs.mtr

    @property
    def info(self):
        """
        :return: string of info of the vehicle
        """
        return "Bounding Box: {}\nPhysics Attrs: {}\nVehicle Status: {}".format(
            self.vinfo.bbox, self.vinfo.attrs, self.status.info)
