#!/usr/bin/env python
"""
Module of Occupancy Grid Map
"""
from typing import Tuple, Union  # pylint: disable=unused-import
from recordclass import recordclass
import numpy as np
from vehicle import Vehicle
from dtype import State, RegionType  # pylint: disable=unused-import


OGMForm = recordclass(  # pylint: disable=invalid-name
    'OGMForm',
    ['wic', 'hic', 'res'])
"""
# wic: width of map in cells, along x axis of vehicle frame.
# hic: height of map in cells, along y axis of vehicle frame.
# res: resolution of the grid map meters/cell
"""


class GridMap(object):
    """
    Represents Occupancy Grid Map
    property form: [width of map in cells, height of map in cells, resolution]
    property origin: state of the vehicle when map generated,
                   indicates the coordinate of the midpoint of the map
    property grid: (width of map in cells, height of map in cells) size matrix,
                 each element represent the RegionType of a point on road.
    property seq: sequence number of each map.
                for checking if current data of map is the newest.
    for more info about the relationship between vehicle frame and map frame, see here:
    https://drive.google.com/open?id=1YqxWZfeGnqf7v9fknMnp6X8nqAHJXwL19CFyHvJvuTU
    """
    def __init__(self, vehicle=None):
        # type: (Vehicle) -> None
        """
        :param vehicle: Vehicle class
        """
        self.seq = None
        self.grid = None
        self.origin = None
        self.vehicle = vehicle
        self.form = OGMForm(wic=600, hic=600, res=0.1)

    def refresh(self, data, seq, origin=None, form=None):
        # type: (Union[list, np.ndarray], int, State, tuple) -> None
        """
        refresh the grid map.
        """
        self.seq = seq
        self.origin = origin if origin else self.vehicle.status.state
        if type(data) is list:
            self.grid = (np.array(data, dtype=np.uint8)
                         .reshape((self.form.wic, self.form.hic)))
        else:
            assert type(data) is np.ndarray
            self.grid = data.astype(np.uint8)
        if form:
            self.form.wic = form[0]
            self.form.hic = form[1]
            self.form.res = form[2]

    @property
    def xy2uv(self):
        """
        transformation matrix from (x, y) in vehicle frame to (u, v) in map frame.
        :return: transformation matrix (3x3)
        """
        return np.array([[0., 1. / self.form.res, self.form.hic / 2.],
                         [1. / self.form.res, 0., self.form.wic / 2.],
                         [0., 0., 1.]])

    def is_free(self, state):
        # type: (State) -> Tuple[bool, int]
        """
        check if the vehicle collides with obstacles
        :param state: State needed to be check
        :return: [False, Grade], if any points are in the OCCUPIED region.
                 [True, Grade], if not. and Grade.
        """
        p2ds = self.vehicle.vinfo.bbox2d_at(state=state)
        theta = -self.origin.rotation.y
        loc = self.origin.location
        trl = np.array([[1., 0., -loc.x], [0., 1., -loc.y], [0., 0., 1.]])
        rot = np.array([[np.cos(theta), -np.sin(theta), 0.],
                        [np.sin(theta), np.cos(theta), 0.],
                        [0., 0., 1.]])
        tf_matrix = np.dot(rot, trl)
        xys = np.dot(tf_matrix, p2ds)
        uvs = np.dot(self.xy2uv, xys).astype(np.int)  # pylint: disable=no-member
        # uvs = np.clip(uvs, 0, min([self.form.wic-1, self.form.hic-1]))
        # if out of grid map, consider it as a collision
        if uvs.max() >= self.form.wic or uvs.min() < 0:
            return False, np.inf
        grades = self.grid[uvs[0], uvs[1]]
        indicator = grades >= RegionType.OCCUPIED.value
        return np.sum(indicator) <= 0, grades.max()
