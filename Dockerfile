FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --fix-missing && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y wget bzip2 ca-certificates curl git vim tmux g++ gcc build-essential cmake checkinstall gfortran libjpeg8-dev libtiff5-dev pkg-config yasm libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libv4l-dev qt5-default libgtk2.0-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev doxygen libflann-dev libboost-all-dev proj-data libproj-dev libyaml-cpp-dev cmake-curses-gui libzmq3-dev freeglut3-dev sudo lsb-release

# Install pybind11
RUN cd / && git clone https://github.com/pybind/pybind11 &&\
    cd pybind11 && git checkout v2.10.0 &&\
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF &&\
    make -j6 && make install

# Install Eigen
RUN cd / && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz &&\
    tar xvzf ./eigen-3.4.0.tar.gz &&\
    cd eigen-3.4.0 &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make install

# Install Python 3.8 and pip
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.8 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install Python packages for FoundationPose
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
RUN pip install scipy joblib scikit-learn ruamel.yaml trimesh pyyaml opencv-python imageio open3d transformations warp-lang einops kornia pyrender

# Install Kaolin
RUN cd / && git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
RUN pip install "cython>=0.29.37"
RUN cd /kaolin && FORCE_CUDA=1 python setup.py develop

# Install nvdiffrast
RUN cd / && git clone https://github.com/NVlabs/nvdiffrast
RUN cd /nvdiffrast && pip install .

ENV OPENCV_IO_ENABLE_OPENEXR=1

# Install additional Python packages
RUN pip install scikit-image meshcat webdataset omegaconf pypng roma seaborn opencv-contrib-python openpyxl wandb imgaug Ninja xlsxwriter timm albumentations xatlas rtree nodejs jupyterlab objaverse g4f ultralytics==8.0.120 pycocotools videoio numba h5py

# Install ROS Noetic
# Add the ROS repository keys
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Install ROS packages
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-catkin \
    ros-noetic-vision-msgs \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

# Install Python dependencies for ROS
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-rosdep \
    python3-catkin-tools \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init
RUN rosdep update

# Create catkin workspace
RUN mkdir -p /root/catkin_ws/src
RUN /bin/bash -c  '. /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so; catkin build'

# Clone and build message and service definitions
RUN /bin/bash -c 'cd /root/catkin_ws/src; \
                  git clone https://github.com/v4r-tuwien/object_detector_msgs.git'
RUN /bin/bash -c 'cd /root/catkin_ws/src; \
                  git clone https://gitlab.informatik.uni-bremen.de/robokudo/robokudo_msgs.git'
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd /root/catkin_ws; catkin build'

# Install Python ROS packages
RUN python3 -m pip install \
    catkin_pkg \
    rospkg

RUN python3 -m pip install \
    git+https://github.com/qboticslabs/ros_numpy.git

# Install mesa-utils for OpenGL support
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    mesa-utils \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV SHELL=/bin/bash
RUN ln -sf /bin/bash /bin/sh

WORKDIR /code

CMD ["python", "/code/foundationpose_ros_wrapper.py"]