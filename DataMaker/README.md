# Dataset Generation

Firstly download the CARLA 0.9.6 from [here](http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz). Extract it and you will get a folder named ***CARLA_0.9.6***.

Secondly, download the additional two maps of CARLA from [here0](http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Town06_0.9.6.tar.gz) and [here1](http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Town07_0.9.6.tar.gz). Put them under the directory ***CARLA_0.9.6/Import***, and run this command to import the maps into CARLA.

```shell
(CARLA_0.9.6)$ ./ImportAssets.sh
```

Thirdly, install the CARLA Python API, make sure you are under the master directory ***CARLA_0.9.6***:

```shell
(CARLA_0.9.6)$ easy_install --user PythonAPI/carla/dist/carla-0.9.6-py2.7-linux-x86_64.egg
```

Fourthly, Install the required python packages:

```shell
~$ pip install --user opencv-python scikit-image pygame pyquaternion reeds_shepp numba numpy matplotlib vispy
```

Fifthly, run this command to start CARLA simulator.

```shell
(CARLA_0.9.6)$ ./CarlaUE4.sh
```

Finally, open another terminal and run following command to generate the parking scenario images. The images are stored in the folder ***DeepPlanning/DataMaker/dataset/inputs***. Please make sure that you are under the directory ***DeepPlanning/DataMaker***:

```shell
(DeepPlanning/DataMaker)$ ./scene_maker/situation_maker.py
```

> If it occurs error: 'RuntimeError: time-out of 10000ms while waiting for the simulator', just run the command again.
>
> If it occurs error: 'RuntimeError: failed to connect to newly created map', just run the command again.

To further generate the ground truth parking path with Orientation-aware Space Exploration (OSE) guided Bi-RRT*, firstly run this command to generate OSE heuristics:

```shell
(DeepPlanning/DataMaker)$ ./run_ose_on_scenes.py
```

Then run this command to generate paths with Bi-RRT*, the paths are stored in folder ***YIPSO_Parking_Planner/YIPS/Dataset/dataset/plans/ose***:

```shell
(DeepPlanning/DataMaker)$ ./run_rrts_on_scenes.py
```



# Configuration Space Visualization

generate configuration space:

```sh
(DeepPlanning/DataMaker)$ ./configuration_space.py
```

visualize the space:

```sh
(DeepPlanning/DataMaker)$ ./visual_space.py
```



# Other Utilities

- generate train.csv and valid.csv for YIPS network training and validation: ***cook.py***
- pick out the scenarios which have no ground truth path: ***failed_seqs.py***
- make different inputs with different encoding methods: ***make_different_inputs.py***
- visualize the scenarios and ground truth: ***visualize.py***
- check if the start and goal configuration is collided or not: ***collision_check_on_tasks.py***