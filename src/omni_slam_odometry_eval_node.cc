#include "ros/odometry_eval.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "omni_slam_odometry_eval_node");

    omni_slam::ros::OdometryEval odometry_eval;
    odometry_eval.Run();

    return 0;
}
