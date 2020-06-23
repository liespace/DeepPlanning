#!/usr/bin/env python
import cv2
import numpy
from typing import Tuple


class OGM(object):
    """class to help task about occupancy grid map"""
    def __init__(self, info):
        """
        :param info: must have these items: {
            "wic": width in cells, along x axis of vehicle frame (Right-handed), number of rows of OGM,
            "hic": height in cells, along y axis of vehicle frame (Right-handed), number of cols of OGM,
            "rmc": meter per cell,
            "lov": length of rectangle region considered as occupied by the vehicle,
            "wov": width of rectangle region considered as occupied by the vehicle,
            "hor": the point with z <= hor is considered as a point of road,
            "hoo": the point with hor < z <= hoo is considered as a point of obstacle,}
        """
        self.info = info

    def ogm_rou_from_pcl(self, cloud):
        # type: (numpy.ndarray) -> numpy.ndarray
        """convert point cloud to occupancy grid map"""
        wic, hic, rmc = self.info['wic'], self.info['hic'], self.info['rmc']
        xy2uv = numpy.array([[0., 1. / rmc, hic / 2.], [1. / rmc, 0., wic / 2.], [0., 0., 1.]])
        x_r = (self.info['lov'] / 2., wic * rmc / 2.)
        y_r = (self.info['wov'] / 2., hic * rmc / 2.)
        z_r = (-self.info['hor'], self.info['hoo'])
        borders = (self.info['hor'], self.info['hoo'])
        # pick up points with rules of x, y range
        xs = numpy.where(numpy.fabs(cloud[:, 0]) < x_r[1])
        cloud = cloud[xs]
        ys = numpy.where(numpy.fabs(cloud[:, 1]) < y_r[1])
        cloud = cloud[ys]
        xs = numpy.where(numpy.fabs(cloud[:, 0]) > x_r[0])
        ys = numpy.where(numpy.fabs(cloud[:, 1]) > y_r[0])
        xy = numpy.union1d(xs, ys)
        cloud = cloud[xy]
        # pick up points with rules of z range
        zs = numpy.where(cloud[:, 2] < z_r[1])
        cloud = cloud[zs]
        zs = numpy.where(cloud[:, 2] > z_r[0])
        cloud = cloud[zs]
        # pick up points with rules of road and obstacle
        ogm = numpy.zeros((wic, hic, len(borders) + 1))
        for i, border in enumerate(borders):
            ks = numpy.where(cloud[:, 2] <= border)
            kxy = cloud[ks]
            kxy[:, 2] = 1
            kuv = numpy.dot(xy2uv, kxy.transpose()).astype(numpy.int)
            ogm[kuv[0, :], kuv[1, :], i] = 1
            cloud = numpy.delete(cloud, ks, 0)
        return ogm.astype(numpy.uint8)

    def refine_ogm_rou(self, ogm_rou, okn=(7, 7), rkn=(30, 30), rt=2, ot=2):
        # type: (numpy.ndarray, Tuple[int, int], Tuple[int, int], int , int) -> numpy.ndarray
        road, obstacle = ogm_rou[:, :, 0], ogm_rou[:, :, 1]
        # handle obstacle points
        for i in range(ot):
            obstacle = self.close_around(obstacle, okn)
        obstacle = self.fill_flood(obstacle)
        # handle road points
        for i in range(rt):
            road = self.close_around(road, rkn)
        road = self.fill_flood(road)
        road = self.open_along(road, rkn)
        not_obstacle = -numpy.array(obstacle.copy(), dtype=numpy.int) + 1
        road = not_obstacle & road
        # # extract unknown points
        not_road = -numpy.array(road.copy(), dtype=numpy.int) + 1
        not_obstacle = -numpy.array(obstacle.copy(), dtype=numpy.int) + 1
        unknown = not_road & not_obstacle
        # Set OGM data
        data = obstacle * 255 + unknown * 127 + road * 1
        return data.astype(numpy.uint8)

    @staticmethod
    def close_around(img, kernel_shape):
        img_o = img.copy()
        h, w = kernel_shape[:2]
        center = (int(h / 2), int(w / 2))
        nut = numpy.zeros((h, w)).astype(numpy.uint8)
        nut[int(h / 2), :] = 255
        for degree in range(0, 180, 6):
            rm = cv2.getRotationMatrix2D(center, degree, 1.0)
            kernel = cv2.warpAffine(nut, rm, (w, h))
            img_o = img_o | cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img_o

    @staticmethod
    def fill_flood(img):
        """Fill Flood"""
        h, w = img.shape[:2]
        img_fill = numpy.zeros_like(img)
        img_fill[1:h - 1, 1:w - 1] = img.copy()[1:h - 1, 1:w - 1]
        mask = numpy.zeros((h + 2, w + 2)).astype(numpy.uint8)
        cv2.floodFill(img_fill, mask, (0, 0), 1)
        img_fill_inv = cv2.bitwise_not(img_fill)
        return img | (img_fill_inv == 255)

    @staticmethod
    def open_along(img, kernel_shape):
        kernel = numpy.ones(kernel_shape).astype(numpy.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
