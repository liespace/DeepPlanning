#!/bin/bash
docker start deep-planning && docker exec -it deep-planning bash -c "cd /home/trouble/DeepPlanning/YIPS/ && tensorboard --logdir ./log/"