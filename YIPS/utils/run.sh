#!/bin/bash
docker start deep-planning && docker exec -it deep-planning bash -c "export CUDA_VISIBLE_DEVICES=0 && cd /home/trouble/DeepPlanning/YIPS/ && python train.py"
#docker cp tensorflow_0112:/ml-rrt/. ~/docker_ws/ml-rrt
