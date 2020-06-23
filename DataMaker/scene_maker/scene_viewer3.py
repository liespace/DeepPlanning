#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

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
import logging
import json
import math
import numpy
import reeds_shepp


def draw_sources_boxes(world, source, scene):
    sx = source.location.x
    sy = source.location.y
    sz = source.location.z
    syw = source.rotation.yaw
    biased_x = scene['biased']['x']
    biased_y = scene['biased']['y']
    biased_yaw = scene['biased']['yaw']
    xs = numpy.array([sx - biased_x, sx + biased_x]).repeat(4)
    ys = numpy.array([sy - biased_y, sy + biased_y] * 2).repeat(2)
    yaws = numpy.array([syw - biased_yaw, syw + biased_yaw] * 4)
    xyy = numpy.array([xs, ys, yaws]).transpose().tolist()
    for x, y, yaw in xyy:
        transform = carla.Transform(carla.Location(x=x, y=y, z=sz), carla.Rotation(yaw=yaw))
        color = carla.Color(r=0, b=255, g=0, a=255)
        text = 'biases'
        draw_box_with_arrow(world, transform, color, text)


def center2rear(node, wheelbase=2.850):
    theta, r = node[2] + numpy.pi, wheelbase / 2.
    node[0] += r * numpy.cos(theta)
    node[1] += r * numpy.sin(theta)
    return tuple(node)


def draw_a_arrow(world, transform, color):
    yaw = math.radians(transform.rotation.yaw)
    world.debug.draw_arrow(begin=transform.location + carla.Location(z=0.0),
                           end=transform.location + carla.Location(x=math.cos(yaw), y=math.sin(yaw), z=0.0),
                           thickness=0.2, arrow_size=0.2,
                           color=color)
    # world.debug.draw_point(location=transform.location + carla.Location(z=0.2),
    #                        size=0.2, color=color)


def draw_box_with_arrow(word, transform, color, text, with_arrow=False):
    word.debug.draw_box(
        box=carla.BoundingBox(transform.location + carla.Location(z=0.05),
                              carla.Vector3D(x=(4.925) / 2., y=(2.116) / 2., z=0.05)),
        rotation=transform.rotation,
        color=color)
    yaw = math.radians(transform.rotation.yaw)
    x, y, yaw = center2rear([transform.location.x, transform.location.y, yaw])
    location = carla.Location(x=x, y=y, z=transform.location.z)
    if with_arrow:
        word.debug.draw_arrow(begin=location + carla.Location(z=0.2),
                              end=location + carla.Location(x=math.cos(yaw), y=math.sin(yaw), z=0.2),
                              thickness=0.1, arrow_size=0.2,
                              color=color)
        # word.debug.draw_string(location=transform.location + carla.Location(z=0.5),
        #                        text=text, color=carla.Color(r=255, b=255, g=255, a=255),
        #                        life_time=100.0, draw_shadow=True)


def draw_box_with_arrow2(word, transform, color, text='', with_arrow=False):
    word.debug.draw_box(
        box=carla.BoundingBox(transform.location + carla.Location(z=0.05),
                              carla.Vector3D(x=0.5 / 2., y=0.5 / 2., z=0.05)),
        rotation=transform.rotation,
        color=color)
    yaw = math.radians(transform.rotation.yaw)
    if with_arrow:
        word.debug.draw_arrow(begin=transform.location + carla.Location(z=0.2),
                              end=transform.location + carla.Location(x=math.cos(yaw), y=math.sin(yaw), z=0.2),
                              thickness=0.1, arrow_size=0.2,
                              color=color)
    # word.debug.draw_string(location=transform.location + carla.Location(z=0.5),
    #                        text=text, color=carla.Color(r=255, b=255, g=255, a=255),
    #                        life_time=100.0, draw_shadow=True)


def lcs2gcs(state, origin):
    xo, yo, ao = origin[0], origin[1], origin[2]
    x = state[0] * numpy.cos(ao) - state[1] * numpy.sin(ao) + xo
    y = state[0] * numpy.sin(ao) + state[1] * numpy.cos(ao) + yo
    a = state[2] + ao
    return x, y, a


def state2transform(state, z=0.):
    return carla.Transform(
        carla.Location(x=state[0], y=state[1], z=z), carla.Rotation(yaw=numpy.degrees(state[2])))


def draw_prediction(world, source, no=0):
    filepath = '{}/{}_inference.txt'.format('scene_maker/scenes', no)
    if not os.path.isfile(filepath):
        print('no prediction: {}'.format(filepath))
        return
    origin = (source['x'], -source['y'], -numpy.radians(source['yaw']))
    pred = numpy.loadtxt(filepath, delimiter=',')
    confidence = 0.5
    for p in pred:
        if p[3] > confidence:
            state = lcs2gcs(p[:3], origin)
            state = (state[0], -state[1], -state[2])
            draw_box_with_arrow2(world, state2transform(state, z=0.3), carla.Color(r=0, b=255, g=0, a=255), text='pred', with_arrow=False)


def draw_path(world, source, no=0):
    filepath = '{}/{}_path.txt'.format('scene_maker/scenes', no)
    if not os.path.isfile(filepath):
        print('no path: {}'.format(filepath))
        return
    origin = (source['x'], -source['y'], -numpy.radians(source['yaw']))
    path = numpy.loadtxt(filepath, delimiter=',')
    path = zip(path[:-1], path[1:])
    for x0, x1 in path:
        states = reeds_shepp.path_sample(x0, x1, 5.0, 0.1)
        for state in states:
            arr = lcs2gcs(state, origin)
            arr = (arr[0], -arr[1], -arr[2])
            draw_box_with_arrow2(world, state2transform(arr, z=0.3), carla.Color(r=0, b=0, g=255, a=127))


def draw_trajectory(world, no=0):
    filepath = '{}/{}_trajectory.txt'.format('scene_maker/scenes', no)
    if not os.path.isfile(filepath):
        print('no trajectory: {}'.format(filepath))
        return
    tj = numpy.loadtxt(filepath, delimiter=',')
    # right handed to left handed
    tj = [(t[0], -t[1], -t[2], t[3], t[4]) for t in tj]
    for p in tj:
        draw_a_arrow(world, state2transform(p[:3], z=0.3), carla.Color(r=0, b=255, g=0, a=127))


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    vehicles_list = []
    depth_sensors = []
    cameras = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()
    bp_star = world.get_blueprint_library().filter("vehicle.lincoln.mkz2017")[0]
    bp_star.set_attribute('color', bp_star.get_attribute('color').recommended_values[-1])
    vehicle_library = world.get_blueprint_library().filter("vehicle.*")
    bp_costars = []
    for bp in vehicle_library:
        # print(bp.get_attribute('number_of_wheels').as_int())
        if bp.get_attribute('number_of_wheels').as_int() >= 4:
            bp_costars.append(bp)

    with open('./scene_maker/script.json') as handle:
        script = json.loads(handle.read())

    number = -1
    while True:
        if number >= len(script['scenes']):
            print ('{}/{} is over size, go with the last one'.format(number, len(script['scenes'])))
            number = len(script['scenes']) - 1
        try:
            scene = script['scenes'][number]
            extras = scene['extras']
            source = scene['source']
            target = scene['target']
            source_point = carla.Transform(
                carla.Location(x=source['x'], y=source['y'], z=source['z'] + scene['biased']['z']),
                carla.Rotation(yaw=source['yaw']))
            target_point = carla.Transform(
                carla.Location(x=target['x'], y=target['y'], z=target['z'] + scene['biased']['z']),
                carla.Rotation(yaw=target['yaw']))
            extra_points = [
                carla.Transform(carla.Location(x=extra['x'], y=extra['y'], z=extra['z'] + scene['biased']['z']),
                                carla.Rotation(yaw=extra['yaw'])) for extra in extras]
            client.load_world(str(scene['map_name']))
            world = client.get_world()

            # --------------
            # Spawn vehicles
            # --------------
            # batch = [carla.command.SpawnActor(bp_star, source_point),
            #          carla.command.SpawnActor(bp_star, target_point)]
            player = world.spawn_actor(bp_star, source_point)
            draw_box_with_arrow(world, target_point, carla.Color(r=255, b=0, g=0, a=255), 'Target')
            draw_box_with_arrow(world, source_point, carla.Color(r=0, b=255, g=0, a=255), 'Source')
            #draw_box_with_arrow(world, source_point, carla.Color(r=0, b=0, g=255, a=255), 'Source')

            base = numpy.array([source['x'], source['y'], numpy.radians(source['yaw'])])
            for i in range(5):
                config = base + (numpy.random.random(3) * 2 - 1) * [2.0, 2.0, numpy.pi/6]
                # draw_box_with_arrow(world, state2transform(config, z=0.3), carla.Color(r=255, b=0, g=0, a=255), 'Biased')

            # draw_sources_boxes(world, source_point, scene)
            batch = []
            for transform in extra_points:  # numpy.random.choice(extra_points, len(extra_points))
                batch.append(carla.command.SpawnActor(numpy.random.choice(bp_costars), transform))

            # batch.append(carla.command.SpawnActor(bp_costar, source_point))
            # batch.append(carla.command.SpawnActor(bp_costar, target_point))

            def processing(photos, method=carla.ColorConverter.Raw):
                def process_img(image):
                    image.convert(method)
                    photos[0] = image

                return process_img

            photo_shot, photo_shot1 = [None], [None]
            spectator = world.get_spectator()
            bp_library = world.get_blueprint_library()
            camera_bp = bp_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '1920')
            camera_bp.set_attribute('image_size_y', '1080')
            camera_bp.set_attribute('gamma', '2.0')
            camera_bp.set_attribute('fov', '90')
            camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=source_point.location.x,
                                                                                 y=source_point.location.y,
                                                                                 z=source_point.location.z + 2),
                                                                  source_point.rotation))
            camera.listen(processing(photo_shot))

            camera_bp1 = bp_library.find('sensor.camera.rgb')
            camera_bp1.set_attribute('gamma', '2.0')
            camera_bp1.set_attribute('fov', '90')
            camera_bp1.set_attribute('image_size_x', '1920')
            camera_bp1.set_attribute('image_size_y', '1920')
            camera1 = world.spawn_actor(camera_bp1, carla.Transform(carla.Location(x=target_point.location.x,
                                                                                  y=target_point.location.y,
                                                                                  z=target_point.location.z + 22),
                                                                   carla.Rotation(pitch=-90)))
            camera1.listen(processing(photo_shot1))

            cameras.append(camera)
            cameras.append(camera1)

            bp_library = world.get_blueprint_library()
            sensor_cfg = script['sensors']
            for spec in sensor_cfg:
                bp = bp_library.find(str(spec['type']))
                bp.set_attribute('role_name', str(spec['id']))
                bp.set_attribute('image_size_x', str(spec['width']))
                bp.set_attribute('image_size_y', str(spec['height']))
                bp.set_attribute('fov', str(spec['fov']))
                bp.set_attribute('sensor_tick', str(spec['sensor_tick']))
                trans = carla.Transform(carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
                                        carla.Rotation(pitch=spec['pitch'], roll=spec['roll'], yaw=spec['yaw']))
                depth_sensors.append(world.spawn_actor(bp, trans, attach_to=player))

            depth_shots = [[None], [None], [None], [None]]
            for i, sen in enumerate(depth_sensors):
                sen.listen(processing(depth_shots[i], method=carla.ColorConverter.Depth))  # LogarithmicDepth

            # weather = carla.WeatherParameters(
            #     cloudyness=50.0,
            #     precipitation=0.0,
            #     sun_altitude_angle=30.0)
            # world.set_weather(weather)
            draw_path(world, source, number)
            # draw_prediction(world, source, number)

            for i, response in enumerate(client.apply_batch_sync(batch)):
                if response.error:
                    logging.error(response.error)
                else:
                    vehicles_list.append(response.actor_id)

            while True:
                spectator.set_transform(carla.Transform(carla.Location(x=target_point.location.x,
                                                                       y=target_point.location.y,
                                                                       z=target_point.location.z + 25),
                                                        carla.Rotation(pitch=-90)))
                world.wait_for_tick()
                cmd = raw_input('On scene {}/{}, Enter N to next/ R to reload settings / '
                                'A number to jump to, -1 means the last: '.format(number, len(script['scenes'])))
                cmd = cmd.strip().lower()
                if cmd == '':
                    continue
                elif cmd == 's':
                    for i in range(5):
                        world.wait_for_tick()
                        photo = photo_shot[0]
                        photo1 = photo_shot1[0]
                    photo.save_to_disk('_out/%08d' % photo.frame_number)
                    photo1.save_to_disk('_out/%08d_1' % photo.frame_number)
                    print('_out/%08d' % photo.frame_number)
                elif cmd == 'ds':
                    for depth in depth_shots:
                        photo = depth[0]
                        photo.save_to_disk('_out/%08d' % photo.frame_number)
                        print('_out/%08d' % photo.frame_number)
                elif cmd == 'n':
                    number += 1
                    break
                elif cmd == 'r':
                    with open('./scene_maker/script.json') as handle:
                        script = json.loads(handle.read())
                    break
                elif cmd[-1].isdigit():
                    number = int(cmd)
                    break
                else:
                    continue

        finally:
            print('\ndestroying %d vehicles' % len(vehicles_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
            vehicles_list[:] = []
            client.apply_batch([carla.command.DestroyActor(x) for x in depth_sensors])
            depth_sensors[:] = []
            for camera in cameras:
                camera.destroy()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
