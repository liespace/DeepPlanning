#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1

docker cp ./src/. tensorflow_0112:/home/trouble/deepway/src
docker cp ./dataset/. tensorflow_0112:/home/trouble/deepway/dataset
docker start tensorflow_0112 && docker exec -i tensorflow_0112 bash -c "cd /home/trouble/deepway/ && python ./src/deepway.py" && docker stop tensorflow_0112

#docker cp tensorflow_0112:/ml-rrt/. ~/docker_ws/ml-rrt
