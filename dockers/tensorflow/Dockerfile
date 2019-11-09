FROM tensorflow/tensorflow:1.12.0-gpu
RUN  aptDeps0='git' && aptDeps1='vim'\
     && apt-get update \
     && apt-get install -y $aptDeps0 \
     && apt-get install -y $aptDeps1 \
     && apt-get clean
