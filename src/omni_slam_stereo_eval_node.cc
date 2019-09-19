#include "ros/stereo_eval.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "omni_slam_stereo_eval_node");

    omni_slam::ros::StereoEval stereo_eval;
    stereo_eval.Run();

    return 0;
}

