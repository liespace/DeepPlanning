FROM tensorflow/tensorflow:1.12.0-gpu

# install packages
RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros1-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO kinetic
# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros-base packages
RUN apt-get update && apt-get install -y \
    ros-kinetic-ros-base=1.3.2-0* \
    && rm -rf /var/lib/apt/lists/*

RUN  apt-get update \
     && apt-get install -q -y git vim iputils-ping libpng16-16 python-tk\
     && apt-get clean \
     && pip install --upgrade pip \
     && pip install --no-cache-dir numba reeds_shepp matplotlib opencv-python scipy dubins typing pygame pyquaternion
