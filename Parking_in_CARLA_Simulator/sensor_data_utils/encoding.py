#!/usr/bin/env python
from sensor_data_utils.ogm_helper import OGM
from sensor_data_utils.image_helper import DepthMap
import numpy as np
import cv2


class Encoder(object):

    def __init__(self):
        self.ogm = OGM({"wic": 600, "hic": 600, "rmc": 0.10,
                        "lov": 4.925 + 0.2, "wov": 2.116 + 0.2,
                        "hor": 0.2, "hoo": 2.0})

    @staticmethod
    def transform_source_target_to_array(source, target):
        return np.array([source.location.x, source.location.y, source.location.z, source.rotation.yaw]), \
               np.array([target.location.x, target.location.y, target.location.z, target.rotation.yaw])

    def make_ogm(self, depthmaps, infos, rotation=(0., 0., 0.), hor=None):
        cloud_vision = []
        for i, img in enumerate(depthmaps):
            cloud_vision.extend(self.convert_array_to_cloud(
                depth_img=self.convert_carla_img_to_array(img),
                info=infos[i], rotation=rotation))
        cloud = []
        cloud.extend(cloud_vision)
        self.ogm.info['hor'] = hor if hor is not None else 0.2
        ogm_raw_pcl = self.ogm.ogm_rou_from_pcl(np.array(cloud))
        ogm_fry_grid = self.ogm.refine_ogm_rou(ogm_raw_pcl)
        return ogm_raw_pcl, ogm_fry_grid

    @staticmethod
    def encode2rgb(grid_map, source, target):
        def carla_transform_to_state(carla_trans):
            return carla_trans.location.x, -carla_trans.location.y, -np.radians(carla_trans.rotation.yaw)

        def gcs2lcs(state, origin):
            xo, yo, ao = origin[0], origin[1], origin[2]
            x = (state[0] - xo) * np.cos(ao) + (state[1] - yo) * np.sin(ao)
            y = -(state[0] - xo) * np.sin(ao) + (state[1] - yo) * np.cos(ao)
            a = state[2] - ao
            return np.array((x, y, a))

        def rectangle(wheelbase=0., width=2.116 + 0.2, length=4.925 + 0.2):  # 2.96, 2.2, 4.925
            return np.array([
                [-(length / 2. - wheelbase / 2.), width / 2.], [length / 2. + wheelbase / 2., width / 2.],
                [length / 2. + wheelbase / 2., -width / 2.], [-(length / 2. - wheelbase / 2.), -width / 2.]])

        def transform(poly, pto):
            pts = poly.transpose()
            xyo = np.array([[pto[0]], [pto[1]]])
            rot = np.array([[np.cos(pto[2]), -np.sin(pto[2])], [np.sin(pto[2]), np.cos(pto[2])]])
            return (np.dot(rot, pts) + xyo).transpose()

        def draw_state(layer, state, value, grid_res=0.1, grid_size=600):
            con = np.floor(
                transform(rectangle(), state) / grid_res + grid_size / 2.).astype(int)
            layer = cv2.fillPoly(layer, [con], value)
            return layer

        ori = carla_transform_to_state(source)
        start = gcs2lcs(carla_transform_to_state(source), ori)
        goal = gcs2lcs(carla_transform_to_state(target), ori)

        loc_layer = np.zeros_like(grid_map, dtype=np.uint8)
        loc_layer = draw_state(loc_layer, np.array(goal), 255)
        loc_layer = draw_state(loc_layer, np.array(start), 127)

        ori_layer = np.zeros_like(grid_map, dtype=np.uint8)
        angle = (goal[2] + np.pi) % (2 * np.pi) - np.pi
        if angle > np.pi:
            print(angle)
        angle = (angle + np.pi) / (2. * np.pi) * 255.
        ori_layer = draw_state(ori_layer, np.array(goal), int(angle))
        ori_layer = draw_state(ori_layer, np.array(start), 0)

        obs_layer = np.multiply(grid_map, loc_layer == 0)

        rgb_array = np.zeros((grid_map.shape[0], grid_map.shape[1], 3), dtype=np.uint8)
        rgb_array[:, :, 0] = obs_layer[:]
        rgb_array[:, :, 1] = loc_layer[:]
        rgb_array[:, :, 2] = ori_layer[:]
        return rgb_array

    def make_ogm_lidar(self, lidar, hor=None):
        self.ogm.info['hor'] = hor if hor is not None else 0.2
        points = np.frombuffer(lidar.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        points = -points
        # we also need to permute x and y
        points = points[..., [1, 0, 2]]
        points[:, 2] += 2.
        print(points.shape)
        ogm_raw = self.ogm.ogm_rou_from_pcl(points)
        ogm_fry = self.ogm.refine_ogm_rou(ogm_raw)
        return ogm_fry

    @staticmethod
    def convert_carla_lidar_to_cloud(lidar, l_z=2):
        lidar_data = np.frombuffer(lidar.raw_data, dtype=np.float32)
        lidar_data = np.reshape(lidar_data, (-1, 3))
        lidar_data = -lidar_data
        lidar_data = lidar_data[..., [1, 0, 2]]
        lidar_data[:, 2] += l_z
        return lidar_data.tolist()

    @staticmethod
    def convert_carla_img_to_array(carla_image):
        bgra_image = np.ndarray(shape=(carla_image.height, carla_image.width, 4),
                                dtype=np.uint8, buffer=carla_image.raw_data)
        scales = np.array([65536.0, 256.0, 1.0, 0]) / (256 ** 3 - 1) * 1000
        depth_image = np.dot(bgra_image, scales).astype(np.float32)
        # actually we want encoding '32FC1' for 'cv2' to 'ros image msg'
        # which is automatically selected by cv bridge with passthrough
        return depth_image

    @staticmethod
    def convert_array_to_cloud(depth_img, info, rotation=(0., 0., 0.)):
        info_copy = info.copy()
        info_copy['pitch'] *= -1
        return DepthMap(info_copy) \
            .set_interface(rotation=(90, 0, 90), scale=(1., -1, 1.)) \
            .set_exterface(rotation=rotation, scale=(1., -1., 1.)) \
            .to_point_cloud(depth_img).tolist()
