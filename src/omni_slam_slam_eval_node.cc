#include "ros/slam_eval.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "omni_slam_slam_eval_node");

    omni_slam::ros::SLAMEval slam_eval;
    slam_eval.Run();

    return 0;
}


