#!/usr/bin/env python
""" This module is copied and modified from agents.navigation.local_planner. """
from collections import deque
import carla
import math
import numpy as np


# ==============================================================================
# -- Global variables and functions --------------------------------------------
# ==============================================================================

class LocalPlanner(object):
    """
    MyController implements the basic behavior of following a provided
    trajectory of waypoints.
    The low-level motion of the vehicle is computed by using two PID controllers
    (for the lateral control and the longitudinal control (cruise speed)).
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)

    def __init__(self, world):
        self.world = world
        self.min_distance = 0.2  # 0.2
        # states init
        self.current_state = None
        self.current_velocity = None
        self.target_waypoint = None

        # queue with tuples of (waypoint, RoadOption)
        self.waypoints_queue = deque(maxlen=600)

    def update(self, current_state, current_velocity):
        def center2rear(node, wheelbase=2.850):
            theta, r = node[2] + np.pi, wheelbase / 2.
            node[0] += r * np.cos(theta)
            node[1] += r * np.sin(theta)
            return tuple(node)

        self.current_state = center2rear(list(current_state))
        self.current_velocity = current_velocity

    def set_waypoints_queue(self, wpts):
        """
        Request new waypoints_queue.

        :param waypoints_queue: [waypoint, target speed in m/s]
        :return:
        """
        [self.waypoints_queue.append(p) for p in wpts]
        for p, _ in list(self.waypoints_queue):
            self.draw_box_with_arrow(self.state2transform(p, z=2.3), carla.Color(r=255, b=0, g=0, a=255), life_time=-1)
        #     self.draw_box(self.state2transform(p, z=0.25), carla.Color(r=255, b=0, g=0, a=255), life_time=1.0)

    @staticmethod
    def state2transform(state, z=0.):
        return carla.Transform(
            carla.Location(x=state[0], y=state[1], z=z), carla.Rotation(yaw=np.degrees(state[2])))

    def yield_control_command(self, _target_waypoint, target_speed):
        v_begin, theta = self.current_state[:2], self.current_state[2]
        v_end = (v_begin[0] + math.cos(theta), v_begin[1] + math.sin(theta))

        v_vec = np.array([v_end[0] - v_begin[0], v_end[1] - v_begin[1]])
        w_vec = np.array([_target_waypoint[0] - v_begin[0], _target_waypoint[1] - v_begin[1]])

        print(v_vec, w_vec)

        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        _alpha = _dot
        _ld = np.hypot(w_vec[0], w_vec[1])
        _R = math.fabs(_ld / (2 * math.sin(_alpha)))
        _L = 2.850
        _dot = math.atan2(_L, _R)

        _cross = np.cross(v_vec, w_vec)
        print(_cross)
        if _cross < 0:
            _dot *= -1.0
        _dot *= 1

        # _dot = -_target_waypoint[3]

        steer = np.clip(_dot / math.radians(70), -1 * 0.6, 1 * 0.6)
        # steer = np.degrees(np.arctan2(2.850 * _dot, 1.)) / 40.

        print('Steer calculation:{}, {}, {}, {}, {}, {}'.format(np.degrees(_alpha), _ld, _R, np.degrees(_dot), _cross,
                                                                steer))

        current_speed = np.linalg.norm(self.current_velocity)
        # accel = (target_speed ** 2. - current_speed ** 2.) / (2 * self.far_from_vehicle(_target_waypoint))
        # if accel > 0:
        #     throttle = np.clip(accel / 6., 0, 1) ** (1. / 4)
        #     brake = 0
        # else:
        #     brake = np.clip(np.fabs(accel) / 6., 0, 1) ** 2
        #     throttle = 0
        if np.fabs(target_speed) > 0.01:
            throttle = 0.2  # 0.35
            brake = 0.
        else:
            throttle = 0.0
            brake = 0.0
        if target_speed < 0:
            reverse = True
        else:
            reverse = False
        hand_brake = False

        print('desired steer: {}, desired throttle: {}/{}->{}, desired brake: {}, reverse: {}'
              .format(steer, throttle, current_speed, target_speed, brake, reverse))

        return carla.VehicleControl(throttle=throttle,
                                    steer=steer,
                                    brake=brake,
                                    hand_brake=hand_brake,
                                    reverse=reverse)

    def far_from_vehicle(self, to):
        here = self.current_state
        return math.hypot(here[0] - to[0], here[1] - to[1])

    def run_step(self):
        """
        """
        print('current_state: {}, current_speed: {}'.format(self.current_state, np.linalg.norm(self.current_velocity)))

        if len(self.waypoints_queue) == 0:
            return carla.VehicleControl(throttle=0,
                                        steer=0,
                                        brake=0,
                                        hand_brake=True,
                                        reverse=False)

        # purge the queue of obsolete waypoints
        max_index = -1

        for i, (waypoint, _) in enumerate(self.waypoints_queue):
            if self.far_from_vehicle(waypoint) < self.min_distance:
                max_index = i

        print('max_index : {}, len: {}'.format(max_index, self.far_from_vehicle(
            self.target_waypoint) if self.target_waypoint else 0.))

        if max_index >= 0:
            for i in range(max_index + 1):
                self.waypoints_queue.popleft()

        if not self.waypoints_queue:
            return None

        # target waypoint
        self.target_waypoint, self._target_speed = self.waypoints_queue[0]
        self.draw_box_with_arrow(self.state2transform(
            self.target_waypoint, z=3.3), color=carla.Color(r=255, b=0, g=0, a=127), life_time=0.01)

        print('target_waypoint: {}, target_speed: {}'.format(self.target_waypoint, self._target_speed))

        # yield command
        return self.yield_control_command(self.target_waypoint, self._target_speed)

    def draw_box_with_arrow(self, transform, color, life_time=0.05):
        yaw = math.radians(transform.rotation.yaw)
        self.world.debug.draw_arrow(begin=transform.location + carla.Location(z=0.0),
                                    end=transform.location + carla.Location(x=math.cos(yaw), y=math.sin(yaw), z=0.0),
                                    thickness=0.12, arrow_size=0.4,
                                    color=color, life_time=life_time)

    def draw_box(self, transform, color, life_time=5.0):
        self.world.debug.draw_box(
            box=carla.BoundingBox(transform.location + carla.Location(z=0.05), carla.Vector3D(x=0.25, y=0.25, z=0.05)),
            rotation=transform.rotation, color=color, life_time=life_time)
