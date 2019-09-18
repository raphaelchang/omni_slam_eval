#ifndef _TRACKER_H_
#define _TRACKER_H_

#include <opencv2/opencv.hpp>
#include "data/frame.h"
#include "data/landmark.h"
#include "odometry/five_point.h"
#include <vector>

namespace omni_slam
{
namespace feature
{

class Tracker
{
public:
    Tracker(const int window_size, const int num_scales, const float delta_pix_err_thresh = 5., const float err_thresh = 20., const int term_count = 50, const double term_eps = 0.01);
    void Init(data::Frame &init_frame);
    int Track(std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, std::vector<double> &errors);

private:
    cv::TermCriteria termCrit_;
    const cv::Size windowSize_;
    const int numScales_;
    const float errThresh_;
    const float deltaPixErrThresh_;
    data::Frame *prevFrame_;
};

}
}
#endif /* _TRACKER_H_ */
