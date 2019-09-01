#include "ros/matching_eval.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "omni_slam_matching_eval_node");

    omni_slam::ros::MatchingEval matching_eval;
    matching_eval.Run();

    return 0;
}
