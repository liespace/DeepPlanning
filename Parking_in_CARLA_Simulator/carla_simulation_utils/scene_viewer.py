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
    sys.path.append(glob.glob('./scene_maker/CARLA/dist/CARLA-*%d.%d-%s.egg' % (
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


def draw_sources_boxes(world, source, scene):
    sx = source.location.x
    sy = source.location.y
    sz = source.location.z
    syw = source.rotation.yaw
    biased_x = scene['biased']['x']
    biased_y = scene['biased']['y']
    biased_yaw = scene['biased']['yaw']
    xs = numpy.array([sx-biased_x, sx+biased_x]).repeat(4)
    ys = numpy.array([sy-biased_y, sy+biased_y] * 2).repeat(2)
    yaws = numpy.array([syw - biased_yaw, syw + biased_yaw] * 4)
    xyy = numpy.array([xs, ys, yaws]).transpose().tolist()
    for x, y, yaw in xyy:
        transform = carla.Transform(carla.Location(x=x, y=y, z=sz), carla.Rotation(yaw=yaw))
        color = carla.Color(r=0, b=255, g=0, a=255)
        text = 'biases'
        draw_box_with_arrow(world, transform, color, text)


def draw_box_with_arrow(word, transform, color, text):
    word.debug.draw_box(box=carla.BoundingBox(transform.location, carla.Vector3D(x=(4.925 + 0.6)/2., y=(2.116 + 0.6)/2., z=0.8)),
                        rotation=transform.rotation,
                        color=color)
    yaw = math.radians(transform.rotation.yaw)
    word.debug.draw_arrow(begin=transform.location, end=transform.location + carla.Location(x=math.cos(yaw), y=math.sin(yaw), z=0),
                          thickness=0.1, arrow_size=0.2,
                          color=color)
    word.debug.draw_string(location=transform.location, text=text, color=carla.Color(r=255, b=255, g=255, a=255),
                           life_time=100.0, draw_shadow=True)


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    vehicles_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()
    bp_star = world.get_blueprint_library().filter("vehicle.lincoln.mkz2017")[0]
    bp_star.set_attribute('color', bp_star.get_attribute('color').recommended_values[1])
    bp_costar = world.get_blueprint_library().filter('vehicle.lincoln.mkz2017')[0]
    bp_costar.set_attribute('color', bp_costar.get_attribute('color').recommended_values[-1])
    with open('../script.json') as handle:
        script = json.loads(handle.read())
    number = -1
    while True:
        if number >= len(script['scenes']):
            print('{}/{} is over size, go with the last one'.format(number, len(script['scenes'])))
            number = len(script['scenes']) - 1
        try:
            scene = script['scenes'][number]
            extras = scene['extras']
            source = scene['source']
            target = scene['target']
            source_point = carla.Transform(carla.Location(x=source['x'], y=source['y'], z=source['z'] + scene['biased']['z']),
                                           carla.Rotation(yaw=source['yaw']))
            target_point = carla.Transform(carla.Location(x=target['x'], y=target['y'], z=target['z'] + scene['biased']['z']),
                                           carla.Rotation(yaw=target['yaw']))
            extra_points = [carla.Transform(carla.Location(x=extra['x'], y=extra['y'], z=extra['z'] + scene['biased']['z']),
                                            carla.Rotation(yaw=extra['yaw'])) for extra in extras]
            client.load_world(str(scene['map_name']))
            world = client.get_world()

            # --------------
            # Spawn vehicles
            # --------------
            # batch = [CARLA.command.SpawnActor(bp_star, source_point),
            #          CARLA.command.SpawnActor(bp_star, target_point)]
            draw_box_with_arrow(world, target_point, carla.Color(r=0, b=0, g=255, a=255), 'Target')
            draw_box_with_arrow(world, source_point, carla.Color(r=255, b=0, g=0, a=255), 'Source')
            draw_sources_boxes(world, source_point, scene)
            batch = []
            for transform in extra_points:
                batch.append(carla.command.SpawnActor(bp_costar, transform))
            batch.append(carla.command.SpawnActor(bp_costar, source_point))
            batch.append(carla.command.SpawnActor(bp_costar, target_point))

            for i, response in enumerate(client.apply_batch_sync(batch)):
                if response.error:
                    logging.error(response.error)
                    print (extra_points[i].location.x, extra_points[i].location.y)
                else:
                    vehicles_list.append(response.actor_id)

            spectator = world.get_spectator()

            while True:
                spectator.set_transform(carla.Transform(carla.Location(x=source_point.location.x,
                                                                       y=source_point.location.y,
                                                                       z=source_point.location.z + 30),
                                                        carla.Rotation(pitch=-90)))
                world.wait_for_tick()
                cmd = input('On scene {}/{}, Enter N to next/ R to reload settings / '
                                'A number to jump to, -1 means the last: '.format(number, len(script['scenes'])))
                cmd = cmd.strip().lower()
                if cmd == '':
                    continue
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


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
