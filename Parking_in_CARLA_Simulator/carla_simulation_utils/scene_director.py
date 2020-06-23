#!/usr/bin/env python
import carla
import json
import logging
import random
import math
import time
import reeds_shepp
from copy import deepcopy


class SceneDirector(object):
    """
    Scene Director:
        1> read script
        2> set scene script
        3> spawn sensors, player (Ego Vehicle) and costars (NPCs).
    """

    def __init__(self, filepath, world, client):
        self.filepath = filepath
        self.world = world
        self.client = client
        self.script = None
        self.auto_bps = None
        self.scene_builder = None
        self.scene_number = None
        self.scene_biases = (-80, 20)

        self.scene = None
        self.map = None
        self.source = None
        self.target = None
        self.extras = []

        self.player = None
        self.sensors = []
        self.costars = []

        # super params
        self.biased_times = 16
        self.shuffle_times = 10
        self.no = -1
        self.seq = 0

    def action(self, scene_number):
        self.scene_number = scene_number
        self.script = self.read_script()
        self._setup_auto_bps()
        self.scene = self.script['scenes'][scene_number]
        self.map = str(self.scene['map_name'])
        self.target = self.extract_target_from_scene()
        self.source = self.extract_source_from_scene()
        self.extras = self.extract_extras_from_scene()
        if self.world.get_map().name is not self.map:
            self._switch_map()
        self.world.set_weather(carla.WeatherParameters.CloudyNoon)  # ClearSunset, ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon
        self.spawn_player()
        self.spawn_sensors()
        self.spawn_extras()

    def extract_target_from_scene(self):
        if self.scene_number >= 77:
            return carla.Transform(
                carla.Location(x=self.scene['target']['x']+self.scene_biases[0],
                               y=self.scene['target']['y']+self.scene_biases[1], z=self.scene['target']['z']),
                carla.Rotation(yaw=self.scene['target']['yaw']))
        return carla.Transform(
            carla.Location(x=self.scene['target']['x'], y=self.scene['target']['y'], z=self.scene['target']['z']),
            carla.Rotation(yaw=self.scene['target']['yaw']))

    def extract_source_from_scene(self, biased=False):
        source = carla.Transform(
            carla.Location(x=self.scene['source']['x'], y=self.scene['source']['y'], z=self.scene['source']['z']),
            carla.Rotation(yaw=self.scene['source']['yaw']))
        if self.scene_number >= 77:
            source = carla.Transform(
                carla.Location(x=self.scene['source']['x']+self.scene_biases[0],
                               y=self.scene['source']['y']+self.scene_biases[1], z=self.scene['source']['z']),
                carla.Rotation(yaw=self.scene['source']['yaw']))
        if biased:
            biased_x = self.scene['biased']['x'] * random.uniform(-1, 1)
            biased_y = self.scene['biased']['y'] * random.uniform(-1, 1)
            biased_yaw = self.scene['biased']['yaw'] * random.uniform(-1, 1)
            source = carla.Transform(
                carla.Location(x=source.location.x + biased_x, y=source.location.y + biased_y, z=source.location.z),
                carla.Rotation(yaw=source.rotation.yaw + biased_yaw))
        return source

    def extract_extras_from_scene(self, biased=False):
        extras = self.scene['extras']
        size = random.randint(1, len(self.extras)) if biased else len(extras)
        random.shuffle(extras) if biased else None
        if self.scene_number >= 77:
            extras = [
                carla.Transform(carla.Location(x=extra['x']+self.scene_biases[0],
                                               y=extra['y']+self.scene_biases[1],
                                               z=extra['z'] + self.scene['biased']['z']),
                                carla.Rotation(yaw=extra['yaw']))
                for extra in extras[0:size]]
        else:
            extras = [
                carla.Transform(carla.Location(x=extra['x'], y=extra['y'], z=extra['z'] + self.scene['biased']['z']),
                                carla.Rotation(yaw=extra['yaw']))
                for extra in extras[0:size]]
        return extras

    def _switch_map(self):
        self.kill_all()
        self.client.load_world(self.map)
        self.world = self.client.get_world()

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

    def read_script(self):
        with open(self.filepath) as handle:
            script = json.loads(handle.read())
        return script

    def _setup_auto_bps(self):
        whitelist = ['vehicle.audi.etron', 'vehicle.chevrolet.impala',
                     'vehicle.dodge_charger.police',
                     'vehicle.tesla.model3']
        bp_library = self.world.get_blueprint_library()
        self.auto_bps = []
        for item in whitelist:
            self.auto_bps.append(bp_library.filter(item)[0])

    def spawn_player(self, transform=None):
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

    def spawn_sensors(self):
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

    def spawn_extras(self):
        self.kill_costars()
        for extra in self.extras:
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
            if npc:
                npc.set_simulate_physics(False)
                self.costars.append(npc)

    def kill_all(self):
        """
        destroy the current ego vehicle and its sensors
        """
        self.kill_sensors()
        self.kill_player()
        self.kill_costars()

    def kill_player(self):
        if self.player:
            self.player.destroy()
            self.player = None

    def kill_sensors(self):
        if self.sensors:
            for sensor in self.sensors:
                sensor.destroy()
            self.sensors[:] = []

    def kill_costars(self):
        if self.costars:
            for npc in self.costars:
                if npc:
                    npc.destroy()
            self.costars[:] = []

    def move_spectator(self):
        spectator = self.world.get_spectator()
        spectator.set_transform(
            carla.Transform(self.player.get_location() + carla.Location(z=23), carla.Rotation(pitch=-90)))

    def draw_target_and_source(self, with_arrow=True):
        self.draw_box_with_arrow(self.source, carla.Color(r=0, b=255, g=0, a=255), 'Source', with_arrow=with_arrow)
        self.draw_box_with_arrow(self.target, carla.Color(r=0, b=0, g=255, a=255), 'Target', with_arrow=with_arrow)

    def draw_box_with_arrow(self, transform, color, text, with_arrow=True, with_text=False):
        self.world.debug.draw_box(box=carla.BoundingBox(transform.location + carla.Location(z=0.3),
                                                        carla.Vector3D(x=(4.925+0.1)/2, y=(2.116+0.1)/2, z=0.05)),
                                  rotation=transform.rotation,
                                  color=color, life_time=0.01, thickness=0.2)
        if with_arrow:
            yaw = math.radians(transform.rotation.yaw)
            self.world.debug.draw_arrow(begin=transform.location + carla.Location(z=0.9),
                                        end=transform.location + carla.Location(x=math.cos(yaw), y=math.sin(yaw), z=0.9),
                                        thickness=0.2, arrow_size=0.3,
                                        color=color, life_time=0.01)
        if with_text:
            self.world.debug.draw_string(location=transform.location + carla.Location(z=1.0),
                                         text=text, color=carla.Color(r=255, b=255, g=255, a=255),
                                         life_time=0.01, draw_shadow=True)

    def draw_arrow(self, transform, color, z=1.0):
        yaw = math.radians(transform.rotation.yaw)
        self.world.debug.draw_arrow(begin=transform.location + carla.Location(z=z),
                                    end=transform.location + carla.Location(x=math.cos(yaw), y=math.sin(yaw), z=z),
                                    thickness=0.12, arrow_size=0.4,
                                    color=color, life_time=0.01)

    def draw_states_with_box_and_arrow(self, states, color=carla.Color(r=255, b=0, g=0, a=255), z=0.06):
        ori = self.carla_transform_to_state(self.source)
        for state in states:
            state = self.lcs2gcs(state, ori)
            state = self.center2center(state)
            trans = self.state2transform(state, z=z)
            self.draw_box_with_arrow(trans, color=color, text='Sample')

    def draw_yips_path(
            self, yips_path,
            sample_color=carla.Color(r=64, b=255, g=0, a=255),
            path_color=carla.Color(r=0, b=255, g=0, a=255)):
        yips = deepcopy(yips_path)
        ori = self.carla_transform_to_state(self.source)
        start = self.center2rear(self.gcs2lcs(self.carla_transform_to_state(self.source), ori))
        goal = self.center2rear(self.gcs2lcs(self.carla_transform_to_state(self.target), ori))
        yips.insert(0, start)
        yips.append(goal)
        self.draw_path(yips, self.source, sample_color=sample_color, path_color=path_color, z=0.25)

    @staticmethod
    def center2center(node, wheelbase=2.850):
        x, y, theta, r = node[0], node[1], node[2], wheelbase / 2.
        x += r * math.cos(theta)
        y += r * math.sin(theta)
        return x, y, node[2]

    @staticmethod
    def center2rear(node, wheelbase=2.850):
        x, y, theta, r = node[0], node[1], node[2] + math.pi, wheelbase / 2.
        x += r * math.cos(theta)
        y += r * math.sin(theta)
        return x, y, node[2]

    @staticmethod
    def lcs2gcs(state, origin):
        xo, yo, ao = origin[0], origin[1], origin[2]
        x = state[0] * math.cos(ao) - state[1] * math.sin(ao) + xo
        y = state[0] * math.sin(ao) + state[1] * math.cos(ao) + yo
        a = state[2] + ao
        return x, y, a

    @staticmethod
    def gcs2lcs(state, origin):
        xo, yo, ao = origin[0], origin[1], origin[2]
        x = (state[0] - xo) * math.cos(ao) + (state[1] - yo) * math.sin(ao)
        y = -(state[0] - xo) * math.sin(ao) + (state[1] - yo) * math.cos(ao)
        a = state[2] - ao
        return x, y, a

    @staticmethod
    def carla_transform_to_state(carla_trans):
        return carla_trans.location.x, -carla_trans.location.y, -math.radians(carla_trans.rotation.yaw)

    @staticmethod
    def state2transform(state, z=0.):
        return carla.Transform(
            carla.Location(x=state[0], y=-state[1], z=z), carla.Rotation(yaw=math.degrees(-state[2])))

    def draw_box(self, transform, color, lift_time=0.01):
        self.world.debug.draw_box(
            box=carla.BoundingBox(transform.location + carla.Location(z=0.00),
                                  carla.Vector3D(x=0.5 / 2., y=0.5 / 2., z=0.05)),
            rotation=transform.rotation,
            color=color, life_time=lift_time)

    def draw_path(self, path, source, lift_time=0.01, z=0.26, with_box=True,
                  sample_color=carla.Color(r=255, b=0, g=0, a=255),
                  path_color=carla.Color(r=0, b=0, g=255, a=255)):
        if with_box:
            self.draw_states_with_box_and_arrow(path[1:-1], color=sample_color, z=z)
        origin = self.carla_transform_to_state(source)
        path = zip(path[:-1], path[1:])
        for x0, x1 in path:
            states = reeds_shepp.path_sample(x0, x1, 5.0, 0.1)
            for state in states:
                arr = self.lcs2gcs(state, origin)
                self.draw_box(self.state2transform(arr, z=z), path_color, lift_time=lift_time)
