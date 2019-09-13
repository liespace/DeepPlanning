#!/bin/bash
docker cp ./src/. tensorflow_0112:/home/trouble/deepway/src
docker cp ./dataset/. tensorflow_0112:/home/trouble/deepway/dataset
docker cp ./dataset/. tensorflow_0112:/home/trouble/deepway/weights
