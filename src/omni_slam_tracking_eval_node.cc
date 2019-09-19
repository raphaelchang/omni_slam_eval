#include "ros/tracking_eval.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "omni_slam_tracking_eval_node");

    omni_slam::ros::TrackingEval<> tracking_eval;
    tracking_eval.Run();

    return 0;
}
