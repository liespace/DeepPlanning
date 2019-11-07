#!/usr/bin/env python
import numpy as np
from copy import deepcopy, copy
from dtype import Location, Rotation, State, C2Go, TreeNode, Velocity, RoadNode, NStatus
from tf.transformations import euler_from_matrix
from anytree import RenderTree


def main():
    x, y, z = 10., 10., 0.
    loc_a = Location(vec=(x, y, z))
    loc_b = Location(vec=(10.0, 0., z))

    vx, vy, vz = 10., 15., 0.0
    vel_a = Velocity(vec=(vx, vy, vz))
    vel_b = Velocity(vec=(10, 10, 10))

    r, p, y = np.radians(0), np.radians(0), np.radians(30)
    r1, p1, y1 = np.radians(0), np.radians(0), np.radians(30)
    rot_a = Rotation(rpy=(r, p, y))
    rot_b = Rotation(rpy=(r1, p1, y1))

    sta_a = State(loc_a, rot_a, vel_a)
    sta_b = State(loc_b, rot_b, vel_b)

    print (sta_a.info)
    print (sta_b.info)
    print (sta_a.transform(state=sta_b, inv=False).info)

    cos_a = C2Go(vec=(1, 2, 1, 1))
    cos_a.reset_vec(vec=(2, 2, 2, 2))

    # roa_a = RoadNode(state=sta_a, status=NodeStatus.FREE, cost2go=cos_a)

    tr_a = TreeNode(name='a', state=sta_a, status=NStatus.STOP,
                    c2get=cos_a, c2go=cos_a)
    tr_b = TreeNode(name='b', state=sta_a, status=NStatus.STOP,
                    c2get=cos_a, c2go=cos_a, parent=tr_a)
    tr_c = TreeNode(name='c', state=sta_a, status=NStatus.STOP,
                    c2get=cos_a, c2go=cos_a, parent=tr_b)
    tr_d = TreeNode(name='d', state=sta_a, status=NStatus.STOP,
                    c2get=cos_a, c2go=cos_a, parent=tr_b)


if __name__ == '__main__':
    main()
