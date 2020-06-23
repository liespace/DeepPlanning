#!/usr/bin/env python
import glob
import os
import sys

try:
    sys.path.append(glob.glob('./scene_maker/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import json
import logging
import random
from runner import CarlaRunner
from archiver import OTArchiver
import math
import time
import numpy as np


class SceneDirector(object):
    """
    Scene Director
    """

    def __init__(self, filepath, world, client):
        self.filepath = filepath
        self.world = world
        self.client = client
        self.script = None
        self.auto_bps = None
        self.scene_builder = None

        self.scene = None
        self.map = None
        self.source = None
        self.target = None
        self.extras = []

        self.player = None
        self.sensors = []
        self.npcs = []

        # super params
        self.biased_times = 1
        self.shuffle_times = 1
        self.no = -1
        self.seq = 0

    def tick(self, ignore=False):
        self.scene = next(self.scene_builder)
        if not ignore:
            self._setup_map()
            self._setup_target()
            self._setup_source()
            self._setup_extras()
            if self.world.get_map().name is not self.map:
                self._switch_map()
            self._spawn_player()
            self._spawn_sensors()
            # self.draw_target_and_source()

    def _setup_map(self):
        self.map = str(self.scene['map_name'])

    def _setup_target(self):
        self.target = carla.Transform(carla.Location(x=self.scene['target']['x'],
                                                     y=self.scene['target']['y'],
                                                     z=self.scene['target']['z']),
                                      carla.Rotation(yaw=self.scene['target']['yaw']))

    def _setup_source(self):
        self.source = carla.Transform(carla.Location(x=self.scene['source']['x'],
                                                     y=self.scene['source']['y'],
                                                     z=self.scene['source']['z']),
                                      carla.Rotation(yaw=self.scene['source']['yaw']))

    def _setup_extras(self):
        self.extras[:] = self.scene['extras']

    def _switch_map(self):
        self.kill_all()
        self.client.load_world(self.map)
        self.world = self.client.get_world()

    def startup(self, seq=0):
        self.seq = seq
        self._read_script()
        self._setup_auto_bps()
        self.scene_builder = self._scene_generator()

    def _scene_generator(self):
        for scene in self.script['scenes']:
            self.no += 1
            yield scene

    def _source_builder(self):
        sources = []
        for i in range(self.biased_times):
            biased_x = self.scene['biased']['x'] if i > 0 else 0.
            biased_y = self.scene['biased']['y'] if i > 0 else 0.
            biased_z = self.scene['biased']['z']
            biased_yaw = self.scene['biased']['yaw'] if i > 0 else 0.
            biased_x *= random.uniform(-1, 1)
            biased_y *= random.uniform(-1, 1)
            biased_yaw *= random.uniform(-1, 1)
            sources.append(carla.Transform(carla.Location(x=self.source.location.x + biased_x,
                                                          y=self.source.location.y + biased_y,
                                                          z=self.source.location.z + biased_z),
                                           carla.Rotation(yaw=self.source.rotation.yaw + biased_yaw)))
        return sources

    def _extras_builder(self):
        extras_es = []
        for i in range(self.shuffle_times):
            scene_size = len(self.extras)
            less = 1
            most = scene_size
            if i == 0:
                size = scene_size
            elif i == 1:
                size = 0
            else:
                size = random.randint(less, most)
            random.shuffle(self.extras)
            extras_es.append([carla.Transform(carla.Location(x=extra['x'],
                                                             y=extra['y'],
                                                             z=extra['z'] + self.scene['biased']['z']),
                                              carla.Rotation(yaw=extra['yaw']))
                              for extra in self.extras[0:size]])
        return extras_es

    def _read_script(self):
        with open(self.filepath) as handle:
            self.script = json.loads(handle.read())

    def _setup_auto_bps(self):
        whitelist = ['vehicle.audi.etron', 'vehicle.chevrolet.impala',
                     'vehicle.dodge_charger.police', 'vehicle.ford.mustang',
                     'vehicle.tesla.model3']
        bp_library = self.world.get_blueprint_library()
        self.auto_bps = []
        for item in whitelist:
            self.auto_bps.append(bp_library.filter(item)[0])

    def _spawn_player(self, transform=None):
        """
        spawns the ego vehicle
        """
        player_script = self.script['player']
        bp = self.world.get_blueprint_library().find(str(player_script['type']))
        bp.set_attribute('role_name', str(player_script['id']))
        bp.set_attribute('color', bp.get_attribute('color').recommended_values[0])
        if self.player:
            self.kill_all()
        if not transform:
            transform = self.source
        self.player = self.world.spawn_actor(bp, carla.Transform(transform.location +
                                                                 carla.Location(z=self.scene['biased']['z']),
                                                                 transform.rotation))
        self.player.set_simulate_physics(True)
        self.player.apply_control(carla.VehicleControl(hand_brake=True))

    def _spawn_sensors(self):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        """
        self.kill_sensors()
        bp_library = self.world.get_blueprint_library()
        sensor_cfg = self.script['sensors']
        for spec in sensor_cfg:
            bp = bp_library.find(str(spec['type']))
            bp.set_attribute('role_name', str(spec['id']))
            if spec['type'].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(spec['width']))
                bp.set_attribute('image_size_y', str(spec['height']))
                bp.set_attribute('fov', str(spec['fov']))
                bp.set_attribute('sensor_tick', str(spec['sensor_tick']))
            elif spec['type'].startswith('sensor.lidar'):
                bp.set_attribute('range', str(spec['range']))
                bp.set_attribute('rotation_frequency', str(spec['rotation_frequency']))
                bp.set_attribute('channels', str(spec['channels']))
                bp.set_attribute('upper_fov', str(spec['upper_fov']))
                bp.set_attribute('lower_fov', str(spec['lower_fov']))
                bp.set_attribute('points_per_second', str(spec['points_per_second']))
                bp.set_attribute('sensor_tick', str(spec['sensor_tick']))
            # create sensor
            trans = carla.Transform(carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
                                    carla.Rotation(pitch=spec['pitch'], roll=spec['roll'], yaw=spec['yaw']))
            self.sensors.append(self.world.spawn_actor(bp, trans, attach_to=self.player))

    def _spawn_npcs(self, extras):
        self.kill_npcs()
        for extra in extras:
            npc = None
            times = 0
            while npc is None and times < 31:
                times += 1
                bp = random.choice(self.auto_bps)
                if times >= 30:
                    logging.warning('use the smallest 4-wheel vehicle')
                    bp = self.world.get_blueprint_library().filter('vehicle.bmw.isetta')[0]
                bp.set_attribute('role_name', 'costar_{}'.format(time.time()))
                bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
                npc = self.world.try_spawn_actor(bp, extra)
            if not npc:
                logging.warning('npc spawn failed')
            self.npcs.append(npc)

    def kill_all(self):
        """
        destroy the current ego vehicle and its sensors
        """
        self.kill_sensors()
        self.kill_player()
        self.kill_npcs()

    def kill_player(self):
        if self.player and self.player.is_alive:
            self.player.destroy()
            self.player = None

    def kill_sensors(self):
        if self.sensors:
            for sensor in self.sensors:
                sensor.destroy()
            self.sensors[:] = []

    def kill_npcs(self):
        if self.npcs:
            for npc in self.npcs:
                if npc:
                    npc.destroy()
            self.npcs[:] = []

    def get_seq(self):
        seq = self.seq
        self.seq += 1
        return seq

    def move_spectator(self):
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(carla.Location(x=self.player.get_location().x,
                                                               y=self.player.get_location().y,
                                                               z=self.player.get_location().z + 23),
                                                carla.Rotation(pitch=-90)))

    def step(self):
        with CarlaRunner(self.world, self.sensors, fps=10, fs=10) as sync_mode:
            for source in self._source_builder():
                self.player.set_transform(source)
                for extras in self._extras_builder():
                    past = time.time()
                    logging.info('cooking {}th example of Scene {}'.format(self.seq, self.no))
                    self.move_spectator()
                    self._spawn_npcs(extras)
                    assert len(self.npcs) == len(extras)
                    # Advance the simulation and wait for the data. Skip n frame.
                    snapshot, lidar = sync_mode.tick(timeout=2.0)
                    # retrieve the data of sensors
                    archiver = OTArchiver()
                    archiver.root = 'lidarset32'
                    seq = self.get_seq()
                    archiver.archive_task(source, self.target, seq)
                    try:
                        hor = self.scene['hor']
                    except KeyError:
                        hor = None
                    ogm = archiver.archive_ogm_lidar(lidar, seq=seq, hor=hor)
                    archiver.encode2rgb(ogm, source, self.target, seq)
                    now = time.time()
                    logging.info('runtime(s): {}'.format(np.round(now - past, 3)))

    def draw_target_and_source(self):
        self.draw_box_with_arrow(self.source, carla.Color(r=255, b=0, g=0, a=255), 'Source')
        self.draw_box_with_arrow(self.target, carla.Color(r=0, b=0, g=255, a=255), 'Target')

    def draw_box_with_arrow(self, transform, color, text):
        self.world.debug.draw_box(box=carla.BoundingBox(transform.location + carla.Location(z=1.0),
                                                        carla.Vector3D(x=2.4, y=1.1, z=0.8)),
                                  rotation=transform.rotation,
                                  color=color)
        yaw = math.radians(transform.rotation.yaw)
        self.world.debug.draw_arrow(begin=transform.location + carla.Location(z=1.0),
                                    end=transform.location + carla.Location(x=math.cos(yaw), y=math.sin(yaw), z=1.0),
                                    thickness=0.1, arrow_size=0.2,
                                    color=color)
        self.world.debug.draw_string(location=transform.location + carla.Location(z=1.0),
                                     text=text, color=carla.Color(r=255, b=255, g=255, a=255),
                                     life_time=100.0, draw_shadow=True)


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    client = carla.Client('localhost', 2000)  # 'localhost' '192.168.32.2
    client.set_timeout(10.0)
    world = client.get_world()
    director = SceneDirector('./scene_maker/lidar_script.json', world=world, client=client)
    director.startup(seq=0)
    try:
        # ignore the scene from here to there
        for j in range(0, 0):
            director.tick(ignore=True)
        # only tick demanded times
        for i in range(0, 87):
            director.tick()
            director.step()
    except StopIteration or IndexError:
        pass
    director.kill_all()
    logging.warn('Have run all scenes')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
