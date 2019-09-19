#include "ros/reconstruction_eval.h"

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "omni_slam_reconstruction_eval_node");

    omni_slam::ros::ReconstructionEval<> reconstruction_eval;
    reconstruction_eval.Run();

    return 0;
}
