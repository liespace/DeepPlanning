#!/usr/bin/env python
from itertools import cycle
import numpy as np
from vispy import app, scene
import vispy.io as vispy_file
from vispy.visuals.transforms import STTransform
from vispy.gloo.util import _screenshot


def main(seq=0):
    # Read volume
    vol1 = np.load('config_space/{}_space.npy'.format(seq))
    vol2 = np.load('config_space/{}_space_1.npy'.format(seq))
    vol = vol1*5 + vol2 * 10
    path = np.load('config_space/{}_path.npy'.format(seq))

    # Prepare canvas
    canvas = scene.SceneCanvas(keys='interactive', size=(1920, 1080), show=True)
    canvas.measure_fps()

    # Set up a viewbox to display the image with interactive pan/zoom
    view = canvas.central_widget.add_view()

    # Set whether we are emulating a 3D texture
    volume1 = scene.visuals.Volume(
        vol, clim=[0., 10.], threshold=0.225,
        emulate_texture=False, relative_step_size=1.5,
        method='iso', parent=view.scene)
    cube1 = scene.visuals.Cube(size=5, color='red', edge_color='black', parent=view.scene)
    cube1.transform = scene.transforms.STTransform(translate=(path[0][2], path[0][1], path[0][0]))
    cube2 = scene.visuals.Cube(size=5, color='green', edge_color='black', parent=view.scene)
    cube2.transform = scene.transforms.STTransform(translate=(path[-1][2], path[-1][1], path[-1][0]))

    # Create camera Arcball
    view.camera = scene.cameras.ArcballCamera(parent=view.scene, fov=0., name='Arcball')
    # Create an XYZAxis visual
    # axis = scene.visuals.XYZAxis(parent=view)
    # s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
    # affine = s.as_matrix()
    # axis.transform = affine

    @canvas.connect
    def on_key_press(ev):
        print(ev.key.name)
        if ev.key.name in 'S':
            print("Saving...")
            res = _screenshot()
            vispy_file.write_png('config_space/{}_shot.png'.format(seq), res)
            print("Done")


if __name__ == '__main__':
    for no in [3520, 4960, 8320, 12320]:
        main(seq=no)
        app.run()
