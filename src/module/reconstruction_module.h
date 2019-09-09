#ifndef _RECONSTRUCTION_MODULE_H_
#define _RECONSTRUCTION_MODULE_H_

#include <vector>
#include <set>
#include <memory>

#include "reconstruction/triangulator.h"
#include "data/landmark.h"

#include <pcl/common/projection_matrix.h>
#include <pcl/point_types.h>

using namespace Eigen;

namespace omni_slam
{
namespace module
{

class ReconstructionModule
{
public:
    struct Stats
    {
    };

    ReconstructionModule(std::unique_ptr<reconstruction::Triangulator> &triangulator);
    ReconstructionModule(std::unique_ptr<reconstruction::Triangulator> &&triangulator);

    void Update(std::vector<data::Landmark> &landmarks);

    Stats& GetStats();
    void Visualize(cv::Mat &img, pcl::PointCloud<pcl::PointXYZRGB> &cloud);

private:
    class Visualization
    {
    public:
        void UpdatePoint(int index, const Vector3d &point);
        void AddPoint(const cv::Point2f &pix);
        void Reserve(int size);
        void OutputPointCloud(cv::Mat &img, pcl::PointCloud<pcl::PointXYZRGB> &cloud);

    private:
        pcl::PointCloud<pcl::PointXYZRGB> cloud_;
        std::vector<cv::Point2f> newPts_;
        int numNewPts_{0};
        std::set<int> goodIndices_;
    };

    std::shared_ptr<reconstruction::Triangulator> triangulator_;

    Stats stats_;
    Visualization visualization_;

    int lastLandmarksSize_{0};
};

}
}

#endif /* _RECONSTRUCTION_MODULE_H_ */
