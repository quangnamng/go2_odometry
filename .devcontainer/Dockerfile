# FROM isaac_ros_dev-aarch64
FROM ros:humble
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
# uodate and install dependencies 
RUN apt-get update && apt-get install -y \
    ros-humble-rmw-cyclonedds-cpp ros-humble-rosidl-generator-dds-idl \
    libyaml-cpp-dev \
    ros-humble-xacro \
    libboost-all-dev\
    build-essential \
    python3-pip \
    cmake \
    libpoco-dev \
    libassimp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cheange the ROS2 RMW to CycloneDDS as instructed by Unitree
RUN cd / && git clone https://github.com/unitreerobotics/unitree_ros2 && cd /unitree_ros2/cyclonedds_ws/src && \
git clone https://github.com/ros2/rmw_cyclonedds -b humble && git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x &&\
cd .. && colcon build --packages-select cyclonedds && source /opt/ros/humble/setup.bash && colcon build
ENV CYCLONEDDS_HOME "/unitree_ros2/cyclonedds_ws/install/cyclonedds"
RUN source /unitree_ros2/cyclonedds_ws/install/setup.bash && cd /root && git clone https://github.com/eclipse-cyclonedds/cyclonedds-python -b releases/0.10.x && cd cyclonedds-python && python3 -m pip install .

# Complile and Install eigenpy
WORKDIR /home
RUN source /unitree_ros2/cyclonedds_ws/install/setup.bash && python3 -m pip install scipy && python3 -m pip install --no-cache-dir eigenpy
RUN git clone https://github.com/stack-of-tasks/eigenpy.git && cd eigenpy && mkdir build && cd build && \
    cmake -DPYTHON_EXECUTABLE='/usr/bin/python3' -DCMAKE_BUILD_TYPE=Release  -DCMAKE_INSTALL_PREFIX=/usr  .. && make -j20 && make install

# Compile and install pinocchio
WORKDIR /home
RUN git clone https://github.com/stack-of-tasks/pinocchio.git && cd pinocchio && mkdir build && cd build && \
    source /unitree_ros2/cyclonedds_ws/install/setup.bash && \ 
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_WITH_COLLISION_SUPPORT=OFF \
          -DBUILD_WITH_OPENMP_SUPPORT=ON \
          -DBUILD_BENCHMARK=OFF \
          -DBUILD_EXAMPLES=OFF \
          -DBUILD_TESTING=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr \
          -DPYTHON_EXECUTABLE='/usr/bin/python3' .. && \
     make -j20 && make install

# Install inekf library
RUN source /opt/ros/humble/setup.bash && cd /home && git clone https://github.com/Rooholla-KhorramBakht/invariant-ekf.git && \
    cd invariant-ekf && mkdir build && cd build && cmake -DPYTHON_EXECUTABLE='/usr/bin/python3' .. && make -j20 && make install


RUN echo 'export PYTHONPATH=$PYTHON_PATH:/home/pinocchio/build/bindings/python' >> ~/.bashrc
RUN echo 'export PYTHONPATH=$PYTHONPATH:/home/invariant-ekf/build/bindings/' >> ~/.bashrc
RUN echo 'source /unitree_ros2/cyclonedds_ws/install/setup.bash' >> ~/.bashrc
RUN mkdir -p /home/ros2_ws/src && cd /home/ros2_ws/src && git clone https://github.com/inria-paris-robotics-lab/unitree_description.git

RUN export PYTHONPATH=$PYTHONPATH:/home/invariant-ekf/build/bindings/ 
RUN cd /home/ros2_ws && colcon build
RUN echo 'source /home/ros2_ws/install/setup.bash' >> ~/.bashrc
RUN source /home/ros2_ws/install/setup.bash
# set the entrypoint to bash