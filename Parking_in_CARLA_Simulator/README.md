# Run YIPSO Parking Path Planning System in CARLA Simulator



## Introduction

This repository is the implementation of running our YIPSO parking path planning system in the [CARLA simulator](http://carla.org/)

<center>
    <img src="../Docs/imgs/YIPSO.png" style="zoom:22%;" title="figure"/><br>
    <p align="left"><b>Figure 1. Our YIPSO Parking Path Planning System.</b> The procedure is straightforward: (1) encoding the scenario as an image, (2) run the neural network YIPS on the image, (3) run the optimizer SO-RRT* on the path from YIPS.</p>
</center>




## Requirements

The code is implemented in **Python 3.6**. We had tested in ***OS Ubuntu 16.04*** with ***NVIDIA Tesla P100*** and ***Intel Xeon E5@3.20GHz***. We recommend you run our code on the similar platform.

You also need following python packages, all you can install with pip:

```shell
~$ pip3 install --user opencv-python scikit-image pygame pyquaternion reeds_shepp numba numpy matplotlib tensorflow==1.12.0
```

## Download Additional Data

Download the additional data from [here](https://github.com/liespace/DeepPlanning/releases/download/v0.1/WeightFilesOfNeuralNetworks.zip). Extract it and you will get a folder named ***weights_of_neural_networks***. Place the folder under the master directory ***Parking_in_CARLA_Simulator***. 



## Setup CARLA Simulator

Firstly download the CARLA 0.9.9 from [here](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.9.tar.gz). Extract it and you will get a folder named ***CARLA_0.9.9***.

Secondly, download the additional maps of CARLA from [here](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.9.tar.gz). Put them under the directory ***CARLA_0.9.9/Import***, and run this command to import the maps into CARLA.

```shell
(CARLA_0.9.9)$ ./ImportAssets.sh
```

Thirdly, install the CARLA Python API, make sure you are under the master directory ***CARLA_0.9.9***:

```shell
(CARLA_0.9.9)$ easy_install --user PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg
```

Finally, run this command to start CARLA simulator.

```shell
(CARLA_0.9.9)$ ./CarlaUE4.sh
```



## Run YIPSO in CARLA Simulator

When the CARLA simulator is running, open another terminal and run this command:

```shell
$ python3 main_loop.py -n 0
```

> ''-n 0" denotes YIPSO parking in the 0th parking scenario. Change the number like "-n 10", then you can park in the 10th parking scenario. The number ranges from 0 to 86.
>
> If it occurs error: 'RuntimeError: time-out of 10000ms while waiting for the simulator', just run the command again.
>
> If it occurs error: 'RuntimeError: failed to connect to newly created map', just run the command again.

Default, the backbone of YIPSO is set to VGG-19. To change the backbone, go to the config.json and change the "WeightName" and set "CheckPoint".  Following backbones are available:

- VGG-19

  ```json
  "WeightName": "rgous-vgg19v2C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr75_steps10[75, 105, 135]_wp0o0e+00)",
  "CheckPoint": 200
  ```

- VGG-16

  ```json
  "WeightName": "rgous-vgg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr70_steps10[70, 95, 110]_wp0o0e+00)",
  "CheckPoint": 200
  ```

- SVG-16 (the VGG-16 backbone of SSD object detector)

  ```json
  "WeightName": "rgous-svg16C-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr1000_steps10[70, 95, 110]_wp0o0e+00)",
  "CheckPoint": 150
  ```

- ResNet-50

  ```json
  "WeightName": "rgous-res50PC-(b16)-(bce_1e+04_1e-04)-(adam_3e-05)-(fr30_steps10[30, 140, 170]_wp0o0e+00)",
  "CheckPoint": 200
  ```