# S2M2 ROS
A ROS node for [s2m2](https://github.com/junhong-3dv/s2m2) stereo depth model.

## Dependencies
- s2m2: setup following instructions this [s2m2 fork](https://github.com/MedericFourmy/s2m2)
- ROS2 installation

## Run node
`ros2 run s2m2_ros s2m2_node`

## Realsense Demo
`ros2 launch realsense2_camera rs_launch.py --enable_infra1 --enable_infra2`  

Higher resolution:  
`ros2 launch realsense2_camera rs_launch.py enable_infra1:=true enable_infra2:=true pointcloud.enable:=true rgb_camera.color_profile:=1280x720x6 depth_module.depth_profile:=1280x720x6 depth_module.infra_profile:=1280x720x6`

## TODO
- launch file + config
- rviz config for demo