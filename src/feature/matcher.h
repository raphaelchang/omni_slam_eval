#ifndef _MATCHER_H_
#define _MATCHER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "data/frame.h"
#include "data/landmark.h"
#include <string>
#include <vector>

namespace omni_slam
{
namespace feature
{

class Matcher
{
public:
    Matcher(std::string descriptor_type, double max_dist = 0);

    std::map<std::pair<int, int>, int> Match(const std::vector<data::Landmark> &train, const std::vector<data::Landmark> &query, std::vector<data::Landmark> &matches) const;

private:
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    double maxDistance_;
};

}
}

#endif /* _MATCHER_H_ */
