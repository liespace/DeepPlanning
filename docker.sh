#!/bin/bash
#docker volume create trouble
#docker volume inspect trouble
#docker rm $(docker ps -aq)
docker run --runtime=nvidia -it --name tensorflow_0112 --mount type=volume,source=trouble,target=/home/trouble -p 8888:6006 tensorflow:1.12.0-gpu-vim-git bash
