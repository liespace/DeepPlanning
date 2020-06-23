#!/usr/bin/env python
import os
import numpy
import math
import cv2
from adacutil.ogm_helper import OGM
from adacutil.image_helper import DepthMap


class OTArchiver(object):

    def __init__(self):
        self.root = './dataset'
        self.ogm = OGM(info={"wic": 600,  "hic": 600, "rmc": 0.10,
                             "lov": 4.925 + 0.2,  "wov": 2.116 + 0.2,
                             "hor": 0.2, "hoo": 2.0})

    def archive_task(self, source, target, seq):
        s = [source.location.x, source.location.y, source.location.z, source.rotation.yaw]
        t = [target.location.x, target.location.y, target.location.z, target.rotation.yaw]
        task = numpy.array([s, t])
        scenes_folder = '{}/scenes'.format(self.root)
        os.makedirs(scenes_folder) if not os.path.isdir(scenes_folder) else None
        numpy.savetxt('{}/{}_task.txt'.format(scenes_folder, seq), task, delimiter=',')

    def archive_ogm(self, depthmaps, infos, seq, rotation=(0., 0., 0.), hor=None):
        cloud_vision = []
        for i, img in enumerate(depthmaps):
            cloud_vision.extend(self.convert_array_to_cloud(depth_img=self.convert_carla_img_to_array(img),
                                                            info=infos[i],
                                                            rotation=rotation))
        cloud = []
        cloud.extend(cloud_vision)

        self.ogm.info['hor'] = hor if hor is not None else 0.2
        ogm_raw = self.ogm.ogm_rou_from_pcl(numpy.array(cloud))
        ogm_fry = self.ogm.refine_ogm_rou(ogm_raw)
        scenes_folder = '{}/scenes'.format(self.root)
        os.makedirs(scenes_folder) if not os.path.isdir(scenes_folder) else None
        cv2.imwrite('{}/{}_raw_pcl.png'.format(scenes_folder, seq), cv2.cvtColor(ogm_raw*255, cv2.COLOR_RGB2BGR))
        cv2.imwrite('{}/{}_gridmap.png'.format(scenes_folder, seq), ogm_fry)
        return ogm_fry

    def archive_ogm_lidar(self, lidar, seq, hor=None):
        self.ogm.info['hor'] = hor if hor is not None else 0.2
        points = numpy.frombuffer(lidar.raw_data, dtype=numpy.dtype('f4'))
        points = numpy.reshape(points, (int(points.shape[0] / 3), 3))
        points = -points
        # we also need to permute x and y
        points = points[..., [1, 0, 2]]
        points[:, 2] += 2.
        print (points.shape)
        ogm_raw = self.ogm.ogm_rou_from_pcl(points)
        ogm_fry = self.ogm.refine_ogm_rou(ogm_raw)
        scenes_folder = '{}/scenes'.format(self.root)
        os.makedirs(scenes_folder) if not os.path.isdir(scenes_folder) else None
        cv2.imwrite('{}/{}_raw_pcl.png'.format(scenes_folder, seq), cv2.cvtColor(ogm_raw*255, cv2.COLOR_RGB2BGR))
        cv2.imwrite('{}/{}_gridmap.png'.format(scenes_folder, seq), ogm_fry)
        return ogm_fry

    @staticmethod
    def convert_carla_lidar_to_cloud(lidar, l_z=2):
        lidar_data = numpy.frombuffer(lidar.raw_data, dtype=numpy.float32)
        lidar_data = numpy.reshape(lidar_data, (-1, 3))
        lidar_data = -lidar_data
        lidar_data = lidar_data[..., [1, 0, 2]]
        lidar_data[:, 2] += l_z
        return lidar_data.tolist()

    @staticmethod
    def convert_carla_img_to_array(carla_image):
        bgra_image = numpy.ndarray(shape=(carla_image.height, carla_image.width, 4),
                                   dtype=numpy.uint8, buffer=carla_image.raw_data)
        scales = numpy.array([65536.0, 256.0, 1.0, 0]) / (256 ** 3 - 1) * 1000
        depth_image = numpy.dot(bgra_image, scales).astype(numpy.float32)
        # actually we want encoding '32FC1' for 'cv2' to 'ros image msg'
        # which is automatically selected by cv bridge with passthrough
        return depth_image

    @staticmethod
    def convert_array_to_cloud(depth_img, info, rotation=(0., 0., 0.)):
        info_copy = info.copy()
        info_copy['pitch'] *= -1
        return DepthMap(info_copy)\
            .set_interface(rotation=(90, 0, 90), scale=(1., -1, 1.))\
            .set_exterface(rotation=rotation, scale=(1., -1., 1.))\
            .to_point_cloud(depth_img).tolist()

    def encode2rgb(self, ogm, source, target, seq):
        wic = self.ogm.info['wic']
        hic = self.ogm.info['hic']
        rmc = self.ogm.info['rmc']
        xy2uv = numpy.array([[0., 1. / rmc, hic / 2.], [1. / rmc, 0., wic / 2.], [0., 0., 1.]])
        us = numpy.array(range(wic) * hic)
        vs = numpy.repeat(numpy.array(range(hic)), wic)
        ps = numpy.ones(wic * hic)
        uv = numpy.stack((us, vs, ps))
        so = numpy.array([source.location.x, -source.location.y, 0] * wic * hic).reshape(-1, 3).transpose()
        to = numpy.array([target.location.x, -target.location.y, 0] * wic * hic).reshape(-1, 3).transpose()
        sy = -math.radians(-source.rotation.yaw)
        ty = -math.radians(-target.rotation.yaw)
        gts = numpy.array([[numpy.cos(sy), -numpy.sin(sy), 0],
                           [numpy.sin(sy), numpy.cos(sy), 0],
                           [0, 0, 1]], dtype=float)
        gtt = numpy.array([[numpy.cos(ty), -numpy.sin(ty), 0],
                           [numpy.sin(ty), numpy.cos(ty), 0],
                           [0, 0, 1]], dtype=float)
        sxy = numpy.dot(numpy.linalg.inv(xy2uv), uv)
        gxy = numpy.dot(numpy.linalg.inv(gts), sxy) + so
        txy = numpy.dot(gtt, gxy - to)
        sus = numpy.where(numpy.fabs(sxy[0, :]) < self.ogm.info['lov'] / 2.)
        svs = numpy.where(numpy.fabs(sxy[1, :]) < self.ogm.info['wov'] / 2.)
        suv = numpy.intersect1d(sus, svs)
        tus = numpy.where(numpy.fabs(txy[0, :]) < self.ogm.info['lov'] / 2.)
        tvs = numpy.where(numpy.fabs(txy[1, :]) < self.ogm.info['wov'] / 2.)
        tuv = numpy.intersect1d(tus, tvs)
        rgb_array = numpy.zeros((wic, hic, 3), dtype=numpy.uint8)
        rgb_array[:, :, 0] = (ogm / 100. * 255.).astype(numpy.int)
        rgb_array[(uv[:, suv][0, :]).astype(numpy.int), (uv[:, suv][1, :]).astype(numpy.int), 1] = 127
        rgb_array[(uv[:, tuv][0, :]).astype(numpy.int), (uv[:, tuv][1, :]).astype(numpy.int), 1] = 255
        yaw = math.radians(-target.rotation.yaw - (-source.rotation.yaw))
        yaw = int((numpy.arctan2(numpy.sin(yaw), numpy.cos(yaw)) + numpy.pi) / (2 * numpy.pi) * 255.)
        rgb_array[(uv[:, tuv][0, :]).astype(numpy.int), (uv[:, tuv][1, :]).astype(numpy.int), 2] = yaw
        inputs_folder = '{}/inputs'.format(self.root)
        os.makedirs(inputs_folder) if not os.path.isdir(inputs_folder) else None
        cv2.imwrite('{}/{}_encoded.png'.format(inputs_folder, seq), cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))

    @staticmethod
    def tf_to_se2frame(vector, origin):
        theta = -math.radians(-origin.rotation.yaw)
        rm = numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0],
                          [numpy.sin(theta), numpy.cos(theta), 0],
                          [0, 0, 1]], dtype=float)
        om = numpy.array([origin.location.x, -origin.location.y, 0])
        return numpy.dot(rm, vector - om)

    @staticmethod
    def tf_to_se2global(vector, origin):
        theta = -math.radians(-origin.rotation.yaw)
        rm = numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0],
                          [numpy.sin(theta), numpy.cos(theta), 0],
                          [0, 0, 1]], dtype=float)
        om = numpy.array([origin.location.x, -origin.location.y, 0])
        return numpy.dot(numpy.linalg.inv(rm), vector) + om

    @staticmethod
    def close_around(img, kernel_shape):
        img_o = img.copy()
        h, w = kernel_shape[:2]
        center = (h / 2, w / 2)
        nut = numpy.zeros((h, w)).astype(numpy.uint8)
        nut[h / 2, :] = 255
        for degree in range(0, 180, 6):
            M = cv2.getRotationMatrix2D(center, degree, 1.0)
            kernel = cv2.warpAffine(nut, M, (w, h))
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
        img_fill_inv = cv2.bitwise_not(img_fill) / 255
        return img | img_fill_inv

    @staticmethod
    def open_along(img, kernel_shape):
        kernel = numpy.ones(kernel_shape).astype(numpy.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
