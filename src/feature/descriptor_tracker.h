#ifndef _DESCRIPTOR_TRACKER_H_
#define _DESCRIPTOR_TRACKER_H_

#include "tracker.h"
#include "matcher.h"
#include "detector.h"
#include "region.h"
#include <opencv2/opencv.hpp>
#include "data/frame.h"
#include "data/landmark.h"
#include "odometry/five_point.h"
#include <vector>

namespace omni_slam
{
namespace feature
{

class DescriptorTracker : public Tracker, public Matcher, public Detector
{
public:
    DescriptorTracker(std::string detector_type, std::string descriptor_type, std::map<std::string, double> det_args, std::map<std::string, double> desc_args, const float match_thresh = 0., const int keyframe_interval = 1);

private:
    int DoTrack(std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, std::vector<double> &errors, bool stereo);
};

}
}
#endif /* _DESCRIPTOR_TRACKER_H_ */
