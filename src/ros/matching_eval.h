#ifndef _MATCHING_EVAL_H_
#define _MATCHING_EVAL_H_

#include "eval_base.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "module/matching_module.h"

namespace omni_slam
{
namespace ros
{

class MatchingEval : public EvalBase
{
public:
    MatchingEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private);
    MatchingEval() : MatchingEval(::ros::NodeHandle(), ::ros::NodeHandle("~")) {}

private:
    void InitPublishers();

    void ProcessFrame(std::unique_ptr<data::Frame> &&frame);
    void GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data);
    bool GetAttributes(std::map<std::string, std::string> &attributes);
    void Visualize(cv_bridge::CvImagePtr &base_img);

    image_transport::Publisher matchedImagePublisher_;

    std::unique_ptr<module::MatchingModule> matchingModule_;

    std::string detectorType_;
    std::string descriptorType_;
};

}
}

#endif /* _MATCHING_EVAL_H_ */
