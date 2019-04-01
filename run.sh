#!/bin/bash

docker cp ~/docker_ws/ml-rrt/. tensorflow_0112:/home/trouble/ml-rrt

docker start tensorflow_0112 && docker exec -i tensorflow_0112 bash -c "cd /home/trouble/ml-rrt/src && python cluster.py" && docker stop tensorflow_0112

#docker cp tensorflow_0112:/ml-rrt/. ~/docker_ws/ml-rrt
