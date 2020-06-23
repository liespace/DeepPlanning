#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.
import os
import json
import carla
import logging
import argparse
import pygame
import numpy as np
from carla_simulation_utils import interface, scene_director, carla_runner
from sensor_data_utils import encoding
from YIPS import model
from SORRT.path_optimizer import optimize_path
from skimage.transform import resize
from skimage.io import imsave
from controller.path_tracker import simple_local_planner
from SSD2D import ssd512


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def reduction(p, discrimination=0.5):
    p = p.reshape((5, 4))
    p = 1. / (1. + np.exp(-p))
    p[:, -2] = p[:, -2] * 2 * np.pi - np.pi
    p[:, :2] = p[:, 0:2] * 60 - 30
    ss = []
    for s in p:
        if s[3] > discrimination:
            ss.append(list(s[:3]))
    return ss


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def convert_carla_image_to_array(image, converter=carla.ColorConverter.Raw):
    image.convert(converter)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def render_carla_image(image, converter, display, ori=(0, 0)):
    image.convert(converter)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, ori)


def save_carla_image(image, converter=carla.ColorConverter.Raw):
    image.convert(converter)
    image.save_to_disk('shots/%08d_top' % image.frame_number)
    print('Saved shots/%08d_top' % image.frame_number)


def save_image(image, name):
    imsave('shots/{}_front.png'.format(name), image)
    print('Saved shots/{}_front'.format(name))


def render_image(image, display, ori=(0, 0)):
    surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    display.blit(surface, ori)


def render_display(display, hud, front_view, bird_view, sensor, text='Testing', wait=1000):
    render_carla_image(front_view, carla.ColorConverter.Raw, display, (0, 0))
    render_carla_image(bird_view, carla.ColorConverter.Raw, display, (610, 0))
    render_image(sensor, display, (0, 410))
    hud.notification(text)
    hud.render(display=display)
    pygame.display.flip()
    pygame.time.wait(wait)


def render_display2(display, hud, front_view, bird_view, sensor, text='Testing', wait=1000):
    render_image(front_view, display, (0, 0))
    render_carla_image(bird_view, carla.ColorConverter.Raw, display, (610, 0))
    render_image(sensor, display, (0, 410))
    hud.notification(text)
    hud.render(display=display)
    pygame.display.flip()
    pygame.time.wait(wait)


def is_parked(director, threshold=0.2):
    loc_now = director.carla_transform_to_state(director.player.get_transform())
    loc_obj = director.carla_transform_to_state(director.target)
    gap = np.sqrt((loc_now[0] - loc_obj[0])**2 + (loc_now[1] - loc_obj[1])**2)
    return True if gap < threshold else False


def do_something(sync_mode, display, director, hud, yips, ssd2d):
    director.move_spectator()
    encoder = encoding.Encoder()

    while not is_parked(director):
        director.source = director.player.get_transform()

        snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
        front_view = back_view if director.player.get_control().reverse else front_view
        rotation = (director.player.get_transform().rotation.roll, -director.player.get_transform().rotation.pitch, 0.)
        hor = None if 'hor' not in director.scene else director.scene['hor']
        raw_pcl, grid_map = encoder.make_ogm(
            depthmaps=[depth_f, depth_r, depth_b, depth_l],
            infos=director.script['sensors'][:], rotation=rotation, hor=hor)
        render_display(display, hud, front_view, bird_view, raw_pcl*255, text='', wait=1000)  # warm up
        scene_type = 'Parking at Semi-Regular Parking Area, Encoding...'
        render_display(display, hud, front_view, bird_view, raw_pcl * 255, text=scene_type, wait=1000)

        scene_img = encoder.encode2rgb(grid_map, director.source, director.target)
        text = 'Encoded The Image of Parking Scene'
        render_display(display, hud, front_view, bird_view, scene_img, text=text, wait=1000)
        text = 'First-Stage: YIPS Network Planning...'
        render_display(display, hud, front_view, bird_view, scene_img, text=text, wait=0)

        input_img = resize(scene_img, (480, 480), order=1) * 255
        yips_path = yips.predict_on_sample(np.array([input_img]))
        yips_path = reduction(yips_path)
        director.draw_yips_path(yips_path)
        director.draw_target_and_source()
        snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
        front_view = back_view if director.player.get_control().reverse else front_view
        front_view = ssd512.ssd_inference(ssd2d, convert_carla_image_to_array(front_view))
        text = 'First-Stage: YIPS Planned Initial Parking Path (Fused with SSD Object Detection)'
        render_display2(display, hud, front_view, bird_view, scene_img, text=text, wait=1000)

        text = 'Second-Stage: SO-RRT* Optimizer Optimizing...'
        render_display2(display, hud, front_view, bird_view, scene_img, text=text, wait=0)

        parking_trajectory, parking_path = optimize_path(director.source, director.target, grid_map, yips_path)
        # director.draw_yips_path(yips_path)
        director.draw_path(parking_path, director.source, lift_time=-1, with_box=False)
        director.draw_target_and_source(with_arrow=False)
        snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
        front_view = back_view if director.player.get_control().reverse else front_view
        front_view = ssd512.ssd_inference(ssd2d, convert_carla_image_to_array(front_view))
        text = 'Second-Stage: SO-RRT* Optimized the Initial Parking Path, Publishing to Controller...'
        render_display2(display, hud, front_view, bird_view, scene_img, text=text, wait=1000)

        save_carla_image(bird_view)
        save_image(front_view, '%08d' % bird_view.frame_number)

        simple_local_planner(
            director, parking_trajectory, sync_mode, render_display, display, hud, scene_img, ssd2d)

    while True:
        director.source = director.player.get_transform()
        control = director.player.get_control()
        control.steer = 0
        control.throttle = 0
        control.reverse = False
        control.hand_brake = True
        director.player.apply_control(control)
        snapshot, depth_f, depth_r, depth_b, depth_l, front_view, bird_view, back_view = sync_mode.tick(timeout=2.0)
        front_view = back_view if director.player.get_control().reverse else front_view
        rotation = (director.player.get_transform().rotation.roll, -director.player.get_transform().rotation.pitch, 0.)
        raw_pcl, grid_map = encoder.make_ogm(
            depthmaps=[depth_f, depth_r, depth_b, depth_l],
            infos=director.script['sensors'][:], rotation=rotation)
        scene_img = encoder.encode2rgb(grid_map, director.source, director.target)
        render_display(display, hud, front_view, bird_view, scene_img, text='Finished', wait=0)


def game_loop(config, scene_number):
    pygame.init()
    pygame.font.init()

    client = carla.Client(config['CARLA']['Host'], config['CARLA']['Port'])
    client.set_timeout(10.0)
    world = client.get_world()
    display = pygame.display.set_mode(
        (config['CARLA']['Window'][0], config['CARLA']['Window'][1]),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

    hud = interface.HUD(config['CARLA']['Window'][0], config['CARLA']['Window'][1])
    # keyboard_controller = interface.KeyboardControl(world, False)
    director = scene_director.SceneDirector(config['Script']['Filepath'], world, client)
    director.action(scene_number)

    weight_folder = config['YIPS']['WeightFolder'] + os.sep + config['YIPS']['WeightName']
    weight_filename = weight_folder + os.sep + 'checkpoint-{}.h5'.format(config['YIPS']['CheckPoint'])
    config_filename = weight_folder + os.sep + 'config.json'
    config = json.loads(open(config_filename).read())
    yips = model.DWModel(config).compile().load_weights(weight_filename)
    yips.predict_on_sample(np.array([np.zeros((480, 480, 3))]))  # warm up
    ssd2d = ssd512.build_ssd512()

    try:
        clock = pygame.time.Clock()
        with carla_runner.CarlaSyncMode(world, director.sensors, fps=100, fs=1) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                do_something(sync_mode, display, director, hud, yips, ssd2d)

                pygame.display.flip()
    finally:
        director.kill_all()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================
def load_configuration(filename='config.json'):
    with open(filename) as handle:
        data = json.loads(handle.read())
    return data


def set_logging_level(config):
    debug, host, port = config['CARLA']['Debug'], config['CARLA']['Host'], config['CARLA']['Port']
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', host, port)


def main():
    argparser = argparse.ArgumentParser(description='YIPSO Planner in CARLA Simulator')
    argparser.add_argument('-n', '--number', default=0, type=int, help='Parking Scenario Number')
    args = argparser.parse_args()

    config = load_configuration()
    set_logging_level(config)
    logging.info('Parking in Scenario {}'.format(args.number))
    try:
        game_loop(config, args.number)
    except KeyboardInterrupt:
        logging.warning('Cancelled by user. Bye!')


if __name__ == '__main__':
    main()
