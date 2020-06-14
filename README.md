# omni_slam_eval
ROS package for my MEng thesis: [Significance of Omnidirectional Fisheye Cameras for Feature-based Visual SLAM](http://raphaelchang.com/MEng_Thesis_Chang.pdf)

This package implements a stereo visual SLAM system for omnidirectional fisheye cameras for the purpose of evaluating the effects of different computer vision algorithms on fisheye-based SLAM. A few changes from traditional SLAM pipelines are introduced, including a novel method for locally rectifying a keypoint patch before descriptor computation for distortion-tolerant feature matching.
