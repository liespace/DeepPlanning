#!/usr/bin/env python
import numpy
from pyquaternion import Quaternion


class DepthMap(object):
    """class to handle processing of depth image"""

    def __init__(self, info):
        # type: (dict) -> None
        """
        :param info: must have these items: {
            "x", "y", "z", # relative to the frame it is in.
            "roll", "pitch", "yaw", # relative to the frame it is in, roll: y->z, pitch: z->x, yaw: x->y
            "width", "height", "fov", "range"}
        """
        self.info = info
        self.intrinsic = self._intrinsic()
        self.extrinsic = self._extrinsic()
        self.interface = numpy.identity(4)
        self.exterface = numpy.identity(4)

    def _intrinsic(self):
        # type: () -> numpy.ndarray
        """
        set the intrinsic matrix
        """
        f = self.info['width'] / 2. / numpy.tan(numpy.radians(self.info['fov'] / 2.))
        return numpy.array([[f,     0.,     self.info['width'] / 2.],
                            [0.,    f,      self.info['height'] / 2.],
                            [0.,    0.,     1.]])

    def _extrinsic(self):
        # type: () -> numpy.ndarray
        """
        set the extrinsic matrix
        """
        return self.get_transform(translation=(self.info['x'], self.info['y'], self.info['z']),
                                  rotation=(self.info['roll'], self.info['pitch'], self.info['yaw']),
                                  scale=(1, 1, 1))

    def set_interface(self, translation=(0., 0., 0.), rotation=(0., 0., 0.), scale=(1., 1., 1.)):
        # type: (tuple, tuple, tuple) -> DepthMap
        """
        set the transform matrix adjust intrinsic frame to extrinsic frame
        :param translation: offset along x, y ,z. (x, y, z)
        :param rotation: rotate along roll, pitch **in degrees**, yaw. (row, pitch, yaw)
        :param scale: scalar of x, y, z. (x, y, z)
        """
        self.interface = self.get_transform(translation, rotation, scale)
        return self

    def set_exterface(self, translation=(0., 0., 0.), rotation=(0., 0., 0.), scale=(1., 1., 1.)):
        # type: (tuple, tuple, tuple) -> DepthMap
        """
        Set the transform matrix adjust extrinsic frame to target frame
        :param translation: offset along x, y ,z. (x, y, z)
        :param rotation: rotate along roll, pitch **in degrees**, yaw. (row, pitch, yaw)
        :param scale: scalar of x, y, z. (x, y, z)
        """
        self.exterface = self.get_transform(translation, rotation, scale)
        return self

    @staticmethod
    def get_transform(translation=(0., 0., 0.), rotation=(0., 0., 0.), scale=(1., 1., 1.)):
        # type: (tuple, tuple, tuple) -> numpy.ndarray
        """
        Set the (right-handed) extrinsic transform matrix
        :param translation: offset along x, y ,z. (x, y, z)
        :param rotation: rotate along roll, pitch **in degrees**, yaw. (row, pitch, yaw)
        :param scale: scalar of x, y, z. (x, y, z)
        """
        qx = Quaternion(axis=[1, 0, 0], angle=numpy.radians(rotation[0]))
        qy = Quaternion(axis=[0, 1, 0], angle=numpy.radians(rotation[1]))
        qz = Quaternion(axis=[0, 0, 1], angle=numpy.radians(rotation[2]))
        q = qz * qy * qx
        transform = q.transformation_matrix
        transform[:3, 3] = translation[:3]
        transform = numpy.dot(transform, numpy.diag([scale[0], scale[1], scale[2], 1.0]))
        return transform

    def to_point_cloud(self, image):
        # type: (numpy.ndarray) -> numpy.ndarray
        """
        convert depth image to point cloud (x, y, z)
        :param image: depth image needed to be converted
        :return: point cloud in form: numpy.array([[x0, y0, z0]...[xn, yn, zn]])
        """
        # reshape to 1-D array
        h, w = image.shape[0], image.shape[1]
        pixels = image.reshape(h * w)
        # structure index matrix
        u = numpy.array(list(range(w)) * h)
        v = numpy.repeat(numpy.array(range(h)), w)
        # find the pixels with depth in expected range and select the objectives
        index = numpy.where(pixels < self.info['range'])
        u = u[index]
        v = v[index]
        pixels = pixels[index]
        # transform from 2d pixels to 3d points
        uv = numpy.array([u, v, numpy.ones_like(u)])
        xy1 = numpy.dot(numpy.linalg.inv(self.intrinsic), uv)
        xyz = xy1 * numpy.vstack((pixels, pixels, pixels))
        # transform the points from image coordinate system to target coordinate system
        xyz1 = numpy.dot(self.interface, numpy.vstack((xyz, numpy.ones_like(pixels))))
        xyz1 = numpy.dot(self.extrinsic, xyz1)
        xyz1 = numpy.dot(self.exterface, xyz1)
        return xyz1[0:3, :].transpose()
