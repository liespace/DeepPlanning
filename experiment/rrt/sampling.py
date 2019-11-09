#!/usr/bin/env python
"""
Sampler Module
"""
from collections import namedtuple
import numpy as np
from dtype import State, Location, Rotation, Velocity
from map import GridMap  # pylint: disable=unused-import


SamplerConfig = namedtuple(
    'SamplerConfig',
    ['r_mean', 'r_sigma', 't_mean', 't_sigma', 'h_mean', 'h_sigma'])
"""
# r_mean:
# r_sigma:
# t_mean:
# t_sigma:
# h_mean:
# h_sigma:
"""

DWSamplerConfig = namedtuple(
    'DWSamplerConfig',
    ['x_mean', 'x_sigma', 'y_mean', 'y_sigma', 't_mean', 't_sigma'])


class GBSE2Sampler(object):
    """
    Sampler Helper Class
    """
    def __init__(self, gridmap=None):
        # type: (GridMap) -> None
        """
        Gaussian Biased Sampler for SE(2) Space.
        """
        self.sample = None
        self.gridmap = gridmap
        self.config = SamplerConfig(r_mean=0., r_sigma=2.,
                                    t_mean=0., t_sigma=np.pi / 4.,
                                    h_mean=0., h_sigma=np.pi / 35.)

    def sampling(self, base):
        # type: (State) -> State
        """
        :param base: biasing base
        :return:
        """
        r = np.random.normal(self.config.r_mean, self.config.r_sigma)
        theta = np.random.normal(self.config.t_mean, self.config.t_sigma)
        heading = np.random.normal(self.config.h_mean, self.config.h_sigma)
        loc = Location(vec=[r * np.cos(theta), r * np.sin(theta), 0])
        rot = Rotation(rpy=(0., 0., heading))
        vel = Velocity(vec=(0., 0., 0.))
        bias = State(location=loc, rotation=rot, velocity=vel)
        sample = base.transform(state=bias, inv=False)
        is_free, _ = self.gridmap.is_free(state=sample)
        self.sample = sample if is_free else None
        return self.sample

    def info(self):
        # type: () -> str
        """
        info of sampler.
        :return: string of info.
        """
        return ('r_mean: {}\nr_sigma: {}\nt_mean: {}\n'
                't_sigma: {}\nh_mean: {}\nh_sigma: {}'.format(
                    self.config.r_mean, self.config.r_sigma,
                    self.config.t_mean, self.config.t_sigma,
                    self.config.h_mean, self.config.h_sigma))


class DWGBSE2Sampler(object):
    """
    Sampler Helper Class
    """
    def __init__(self, gridmap=None):
        # type: (GridMap) -> None
        """
        DeepWay Gaussian Biased Sampler for SE(2) Space.
        """
        self.sample = None
        self.gridmap = gridmap
        self.config = DWSamplerConfig(x_mean=0., x_sigma=2.,
                                      y_mean=0., y_sigma=2.,
                                      t_mean=0., t_sigma=0.4)

    def sampling(self, base):
        # type: (State) -> State
        """
        :param base: biasing base
        :return:
        """
        x = np.random.normal(self.config.x_mean, self.config.x_sigma)
        y = np.random.normal(self.config.y_mean, self.config.y_sigma)
        t = np.random.normal(self.config.t_mean, self.config.t_sigma)
        loc = Location(vec=[x, y, 0])
        rot = Rotation(rpy=(0., 0., t))
        vel = Velocity(vec=(0., 0., 0.))
        bias = State(location=loc, rotation=rot, velocity=vel)
        sample = base.transform(state=bias, inv=False)
        is_free, _ = self.gridmap.is_free(state=sample)
        self.sample = sample if is_free else None
        return self.sample

    def info(self):
        # type: () -> str
        """
        info of sampler.
        :return: string of info.
        """
        return ('x_mean: {}\nx_sigma: {}\ny_mean: {}\n'
                'y_sigma: {}\nt_mean: {}\nt_sigma: {}'.format(
                    self.config.x_mean, self.config.x_sigma,
                    self.config.y_mean, self.config.y_sigma,
                    self.config.t_mean, self.config.t_sigma))
