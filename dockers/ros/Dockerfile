FROM osrf/ros:kinetic-desktop-full-xenial
RUN  aptDeps0='git' && aptDeps1='vim'\
     && apt-get update \
     && apt-get install -y $aptDeps0 \
     && apt-get install -y $aptDeps1 \
     && apt-get clean
