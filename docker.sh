#!/bin/bash
#docker volume create trouble
#docker volume inspect trouble
#docker rm $(docker ps -aq)
docker run --runtime=nvidia -it --mount type=volume,source=trouble,target=/notebooks -p 8888:8888 trouble/tensorflow:1.12.0-gpu-py3-git bash
