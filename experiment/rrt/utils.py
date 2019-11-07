#!/usr/bin/env python
from autopilot_msgs.msg import WayPoints, RouteNode
from anytree.iterators import PreOrderIter
from anytree.search import findall
from numpy.linalg import norm
from dtype import TreeNode, State, Point3D, Posture, GeoReference
from planner import Planner
from std_msgs.msg import Header, ColorRGBA
from selection import Way
from typing import List
from visualization_msgs.msg import Marker, MarkerArray
import geometry_msgs.msg as gmm
import numpy as np
import pyproj as pj
import rospy

ij = pj.Proj('+proj=tmerc +lat_0=49 +lon_0=8 +k=1 +x_0=0 +y_0=0 '
             '+datum=WGS84 +units=m +geoidgrids=egm96_15.gtx +vunits=m +no_defs')
oj = pj.Proj(init='epsg:4326')


def msg_way(way=None):
    # type: (Way) -> WayPoints
    """
    convert Way to WayPoints of autopilot_msgs.msg
    :return: WayPoints object
    """
    return WayPoints(points=[RouteNode(x=node.state.position.x,
                                       y=node.state.position.y,
                                       latitude=pj.transform(ij, oj, node.state.position.x, node.state.position.y)[0],
                                       longitude=pj.transform(ij, oj, node.state.position.x, node.state.position.y)[1])
                             for node in way.way],
                     speeds=way.speeds,
                     header=Header(stamp=rospy.Time.now()))


def msg_tree_marker(root=None, size=500):
    # type: (TreeNode, int) -> MarkerArray
    """
    convert tree to MarkerArray
    :param root: root node of Tree
    :param size: expected maximum size of tree
    :return: visualization_msgs/MarkerArray
    """
    namespace = '/rrt_mp/tree'
    frame_id = 'map'
    max_size = size * 2

    pts = list(np.array([[node.parent.state.position.to_ros_point(), node.state.position.to_ros_point()]
                         for node in findall(root, filter_=lambda n: n.parent is not None)]).flatten())
    for pt in pts:
        pt.z = 0.4
    lines = Marker(header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                   ns=namespace,
                   id=0,
                   type=Marker.LINE_LIST,
                   action=Marker.ADD,
                   scale=gmm.Vector3(x=0.1),
                   color=ColorRGBA(a=1., r=0.5, g=0.25, b=0.25),
                   points=pts)

    preorder = list(PreOrderIter(root))
    if len(preorder) > max_size:
        rospy.logwarn("TS is over, visualization isn't completed, TS {}, MAX {}".format(len(preorder), max_size))

    arrows = [Marker(header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                     ns=namespace,
                     id=i + 1,
                     type=Marker.ARROW,
                     action=Marker.DELETE) for i in range(max_size)]
    for i, node in enumerate(preorder):
        arrows[i].action = Marker.ADD
        arrows[i].scale = gmm.Vector3(x=0.5, y=0.1, z=0.1)
        arrows[i].color = ColorRGBA(a=1.0, r=0.5, g=0.25, b=0.25)
        pose = node.state.to_ros_pose()
        pose.position.z = 0.4
        arrows[i].pose = pose

    markers = MarkerArray()
    markers.markers.append(lines)
    markers.markers.extend(arrows)
    return markers


def cross_state(start=None, end=None):
    # type: (State, State) -> State
    """
    generate the state contains two information:
    1. the middle point between start and end.
    2. the orientation of the vector from start to end.
    :param start: start State, only its position attar is used
    :param end: end State, only its position attar is used
    :return: State
    """
    xy = (end.position.array - start.position.array)[:-1]
    return State(position=Point3D((start.position.array + end.position.array) / 2),
                 orientation=Posture(rpy=(0., 0., np.arctan2(xy[1], xy[0]))))


def norm_between_states(start=None, end=None):
    # type: (State, State) -> float
    """
    :param start: start State, only its position attar is used
    :param end: end State, only its position attar is used
    :return: the distance between two states
    """
    return norm((end.position.array - start.position.array)[:-1])


def msg_path_marker(path=None, size=10):
    # type: (List[State], int) -> MarkerArray
    """
    convert path to MarkerArray
    :param path: path made up with States
    :param size: expected maximum size of path
    :return: visualization_msgs/MarkerArray
    """
    namespace = '/rrt_mp/path'
    frame_id = 'map'
    max_size = size * 3
    if len(path) * 3 > max_size:
        rospy.logwarn('Path size is over, visualized path is not completed')

    markers = [Marker(header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                      ns=namespace,
                      id=i,
                      action=Marker.DELETE) for i in range(max_size)]
    # cylinders
    for i, state in enumerate(path):
        pose = state.to_ros_pose()
        pose.position.z = 0.1
        markers[i] = Marker(header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                            ns=namespace,
                            id=i,
                            type=Marker.CYLINDER,
                            action=Marker.ADD,
                            scale=gmm.Vector3(x=2., y=2., z=0.1),
                            color=ColorRGBA(a=0.5, r=0.0, g=0.0, b=0.0),
                            pose=pose)

    # arrow
    for i, state in enumerate(path):
        m = i + len(path)
        pose = state.to_ros_pose()
        pose.position.z = 0.1
        markers[m] = Marker(header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                            ns=namespace,
                            id=m,
                            type=Marker.ARROW,
                            action=Marker.ADD,
                            scale=gmm.Vector3(x=2., y=0.25, z=0.25),
                            color=ColorRGBA(a=0.5, r=0.0, g=0.0, b=0.0),
                            pose=pose)

    # cubes
    for i in range(len(path) - 1):
        n = i + len(path) * 2
        pose = cross_state(path[i], path[i + 1]).to_ros_pose()
        pose.position.z = 0.1
        markers[n] = Marker(header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                            ns=namespace,
                            id=n,
                            type=Marker.CUBE,
                            action=Marker.ADD,
                            scale=gmm.Vector3(x=norm_between_states(path[i], path[i + 1]), y=2.0, z=0.1),
                            color=ColorRGBA(a=0.5, r=0.0, g=0.0, b=0.0),
                            pose=pose)
    return MarkerArray(markers=markers)


def msg_way_marker(way=None, size=100):
    # type: (Way, int) -> MarkerArray
    """
    convert Way to MarkerArray
    :param way: Way()
    :param size: expected maximum size of way
    :return: visualization_msgs/MarkerArray
    """
    namespace = '/rrt_mp/way'
    frame_id = 'map'
    max_size = size * 2
    if len(way.way) > max_size:
        rospy.logwarn('Way size is over, visualized way is not completed')

    cubes = [Marker(header=Header(frame_id=frame_id, stamp=rospy.Time.now()),
                    ns=namespace,
                    id=i,
                    type=Marker.CUBE,
                    action=Marker.DELETE) for i in range(max_size)]
    if way.speeds:
        max_speed = max(way.speeds) + 1e-10
    else:
        max_speed = 1e-10
    for i, node in enumerate(way.way):
        cubes[i].action = Marker.ADD
        cubes[i].scale = gmm.Vector3(x=5, y=2, z=0.1)
        cubes[i].color = ColorRGBA(a=1.0, r=way.speeds[i]/max_speed, g=0.0, b=1-way.speeds[i]/max_speed)
        pose = node.state.to_ros_pose()
        pose.position.z = 0.2
        cubes[i].pose = pose

    return MarkerArray(markers=cubes)


def msg_log(p=None):
    # type: (Planner) -> str
    """
    convert Planner to Str
    :param p: motion planner
    :return: string of info need to be publish
    """
    return '%s | vt@ %.2f | mcr@ %s' \
           '\nPosture: x@ %.2f, y@ %.2f, yaw@ %.2f' \
           '\nVelocity: speed@ %.2f | x@ %.2f | y@ %.2f' \
           '\nAccelerate: x@ %.2f | y@ %.2f | z@ %.2f' % (
        p.selector.way.status, p.selector.way.vt, p.selector.way.mcr,
        p.gridmap.origin.position.x, p.gridmap.origin.position.y, p.gridmap.origin.orientation.yaw,
        p.vehicle.status.velocity().speed, p.vehicle.status.velocity().x, p.vehicle.status.velocity().y,
        p.vehicle.status.accel.linear.x, p.vehicle.status.accel.linear.y, p.vehicle.status.accel.linear.z)
