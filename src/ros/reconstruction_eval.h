#ifndef _RECONSTRUCTION_EVAL_H_
#define _RECONSTRUCTION_EVAL_H_

#include "tracking_eval.h"
#include <ros/ros.h>
#include <vector>

#include "module/reconstruction_module.h"

namespace omni_slam
{
namespace ros
{

class ReconstructionEval : public TrackingEval
{
public:
    ReconstructionEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private);
    ReconstructionEval() : ReconstructionEval(::ros::NodeHandle(), ::ros::NodeHandle("~")) {}

private:
    void InitPublishers();

    void ProcessFrame(std::unique_ptr<data::Frame> &&frame);
    void GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data);
    void Visualize(cv_bridge::CvImagePtr &base_img);

    ::ros::Publisher pointCloudPublisher_;

    std::unique_ptr<module::ReconstructionModule> reconstructionModule_;
};

}
}

#endif /* _RECONSTRUCTION_EVAL_H_ */
