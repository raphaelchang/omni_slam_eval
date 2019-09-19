#include "lk_stereo_matcher.h"

#include <cmath>

namespace omni_slam
{
namespace stereo
{

LKStereoMatcher::LKStereoMatcher(double epipolar_thresh, int window_size, int num_scales, float err_thresh, int term_count, double term_eps)
    : StereoMatcher(epipolar_thresh),
    windowSize_(window_size / pow(2, num_scales), window_size / pow(2, num_scales)),
    numScales_(num_scales),
    errThresh_(err_thresh),
    termCrit_(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, term_count, term_eps)
{
}

void LKStereoMatcher::FindMatches(const cv::Mat &image1, const cv::Mat &image2, const std::vector<cv::KeyPoint> &pts1, std::vector<cv::KeyPoint> &pts2, std::vector<int> &matchedIndices) const
{
    std::vector<cv::Point2f> pointsToTrack;
    pointsToTrack.reserve(pts1.size());
    for (const cv::KeyPoint &kpt : pts1)
    {
        pointsToTrack.push_back(kpt.pt);
    }
    std::vector<cv::Point2f> results;
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(image1, image2, pointsToTrack, results, status, err, windowSize_, numScales_, termCrit_, 0);
    pts2.clear();
    matchedIndices.clear();
    for (int i = 0; i < results.size(); i++)
    {
        cv::KeyPoint kpt(results[i], pts1[i].size);
        pts2.push_back(kpt);
        if (status[i] == 1 && err[i] <= errThresh_)
        {
            matchedIndices.push_back(i);
        }
    }
}

}
}
