#!/usr/bin/env python
import pyproj
from typing import Tuple


class GCSHelper(object):
    """class to help stuffs about projection between local coordinate and Geographic Coordinate System"""

    def __init__(self, local_origin=(49., 8.)):
        # type: (Tuple[float, float]) -> None
        """
        lcs: Local Coordinate System (LCS).
        gcs: Geographic Coordinate System (GCS).
        :param local_origin: (latitude, longitude), tuple of the coordinate of the origin of LCS in GCS.
        """
        self.lco = local_origin
        self.lcs = self._local_coordinate_system()
        self.gcs = self._geographic_coordinate_system()

    def _local_coordinate_system(self):
        return pyproj.Proj('+proj=tmerc +lat_0={} +lon_0={} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m '
                           '+geoidgrids=egm96_15.gtx +vunits=m +no_defs'.format(self.lco[0], self.lco[1]))

    @staticmethod
    def _geographic_coordinate_system():
        return pyproj.Proj(init='epsg:4326')

    def to_gcs(self, x, y):
        # type: (float, float) -> Tuple[float, float]
        """project (x, y) in LCS to (lat, lon) in GCS"""
        latitude, longitude = pyproj.transform(self.lcs, self.gcs, x, y)
        return latitude, longitude

    def to_lcs(self, latitude, longitude):
        # type: (float, float) -> Tuple[float, float]
        """project (lat, lon) in GCS to (x, y) in LCS"""
        x, y = pyproj.transform(self.gcs, self.lcs, latitude, longitude)
        return x, y
