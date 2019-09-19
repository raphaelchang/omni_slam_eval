#ifndef _LK_STEREO_MATCHER_H_
#define _LK_STEREO_MATCHER_H_

#include "stereo_matcher.h"

namespace omni_slam
{
namespace stereo
{

class LKStereoMatcher : public StereoMatcher
{
public:
    LKStereoMatcher(double epipolar_thresh, int window_size, int num_scales, float err_thresh = 20., int term_count = 50, double term_eps = 0.01);

private:
    void FindMatches(const cv::Mat &image1, const cv::Mat &image2, const std::vector<cv::KeyPoint> &pts1, std::vector<cv::KeyPoint> &pts2, std::vector<int> &matchedIndices) const;

    cv::TermCriteria termCrit_;
    const cv::Size windowSize_;
    const int numScales_;
    const float errThresh_;
};

}
}

#endif /* _LK_STEREO_MATCHER_H_ */
