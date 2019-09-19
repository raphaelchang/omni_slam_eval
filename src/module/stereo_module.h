#ifndef _STEREO_MODULE_H_
#define _STEREO_MODULE_H_

#include <vector>
#include <set>
#include <memory>

#include "data/landmark.h"
#include "stereo/stereo_matcher.h"

#include <pcl/common/projection_matrix.h>
#include <pcl/point_types.h>

using namespace Eigen;

namespace omni_slam
{
namespace module
{

class StereoModule
{
public:
    struct Stats
    {
    };

    StereoModule(std::unique_ptr<stereo::StereoMatcher> &stereo);
    StereoModule(std::unique_ptr<stereo::StereoMatcher> &&stereo);

    void Update(data::Frame &frame, std::vector<data::Landmark> &landmarks);

    Stats& GetStats();
    void Visualize(cv::Mat &base_img, const cv::Mat &base_stereo_img);

private:
    class Visualization
    {
    public:
        void Init(cv::Size img_size);
        void AddMatch(cv::Point2f pt1, cv::Point2f pt2, double depth, double depthGnd);
        void Draw(cv::Mat &img, const cv::Mat &stereo_img);

    private:
        cv::Mat curMask_;
        cv::Mat curDepth_;
        const double maxDepth_{50.};
    };

    std::shared_ptr<stereo::StereoMatcher> stereo_;

    int frameNum_{0};

    Stats stats_;
    Visualization visualization_;
};

}
}

#endif /* _STEREO_MODULE_H_ */
