#ifndef _LK_TRACKER_H_
#define _LK_TRACKER_H_

#include "tracker.h"
#include <opencv2/opencv.hpp>
#include "data/frame.h"
#include "data/landmark.h"
#include "odometry/five_point.h"
#include <vector>

namespace omni_slam
{
namespace feature
{

class LKTracker : public Tracker
{
public:
    LKTracker(const int window_size, const int num_scales, const float delta_pix_err_thresh = 5., const float err_thresh = 20., const int keyframe_interval = 1, const int term_count = 50, const double term_eps = 0.01);

private:
    int DoTrack(std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, std::vector<double> &errors, bool stereo);

    cv::TermCriteria termCrit_;
    const cv::Size windowSize_;
    const int numScales_;
    const float errThresh_;
    const float deltaPixErrThresh_;
};

}
}
#endif /* _LK_TRACKER_H_ */
