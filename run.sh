#!/bin/bash

docker cp ~/ml-rrt/src/. tensorflow_0112:/home/trouble/ml-rrt/src

docker start tensorflow_0112 && docker exec -i tensorflow_0112 bash -c "cd /home/trouble/ml-rrt/src && python main.py" && docker stop tensorflow_0112

#docker cp tensorflow_0112:/ml-rrt/. ~/docker_ws/ml-rrt
