#!/bin/bash
docker start tensorflow_0112 && docker exec -i tensorflow_0112 bash -c "export CUDA_VISIBLE_DEVICES=1 && cd /home/trouble/deepway/ && python ./src/deepway.py" && docker stop tensorflow_0112
#docker cp tensorflow_0112:/ml-rrt/. ~/docker_ws/ml-rrt
