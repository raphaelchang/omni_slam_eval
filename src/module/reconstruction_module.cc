#include "reconstruction_module.h"

using namespace std;

namespace omni_slam
{
namespace module
{

ReconstructionModule::ReconstructionModule(std::unique_ptr<reconstruction::Triangulator> &triangulator, std::unique_ptr<optimization::BundleAdjuster> &bundle_adjuster)
    : triangulator_(std::move(triangulator)),
    bundleAdjuster_(std::move(bundle_adjuster))
{
}

ReconstructionModule::ReconstructionModule(std::unique_ptr<reconstruction::Triangulator> &&triangulator, std::unique_ptr<optimization::BundleAdjuster> &&bundle_adjuster)
    : ReconstructionModule(triangulator, bundle_adjuster)
{
}

void ReconstructionModule::Update(std::vector<data::Landmark> &landmarks)
{
    if (landmarks.size() == 0)
    {
        return;
    }

    triangulator_->Triangulate(landmarks);

    visualization_.Reserve(landmarks.size());
    for (int i = lastLandmarksSize_; i < landmarks.size(); i++)
    {
        cv::Point2f pt = landmarks[i].GetObservations()[0].GetKeypoint().pt;
        visualization_.AddPoint(pt);
    }
    for (int i = 0; i < landmarks.size(); i++)
    {
        if (landmarks[i].HasEstimatedPosition())
        {
            visualization_.UpdatePoint(i, landmarks[i].GetEstimatedPosition());
        }
    }
    lastLandmarksSize_ = landmarks.size();
}

void ReconstructionModule::BundleAdjust(std::vector<data::Landmark> &landmarks, const std::vector<int> &frame_ids)
{
    bundleAdjuster_->Optimize(landmarks, frame_ids);

    for (int i = 0; i < lastLandmarksSize_; i++)
    {
        if (landmarks[i].HasEstimatedPosition())
        {
            visualization_.UpdatePoint(i, landmarks[i].GetEstimatedPosition());
        }
    }
}

void ReconstructionModule::BundleAdjust(std::vector<data::Landmark> &landmarks)
{
    std::vector<int> temp;
    BundleAdjust(landmarks, temp);
}

ReconstructionModule::Stats& ReconstructionModule::GetStats()
{
    return stats_;
}

void ReconstructionModule::Visualize(cv::Mat &img, pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    visualization_.OutputPointCloud(img, cloud);
}

void ReconstructionModule::Visualization::UpdatePoint(int index, const Vector3d &point)
{
    cloud_.at(index).x = point(0);
    cloud_.at(index).y = point(1);
    cloud_.at(index).z = point(2);
    goodIndices_.insert(index);
}

void ReconstructionModule::Visualization::AddPoint(const cv::Point2f &pix)
{
    pcl::PointXYZRGB pt(0, 0, 0);
    cloud_.push_back(pt);
    newPts_.push_back(pix);
    numNewPts_++;
}

void ReconstructionModule::Visualization::Reserve(int size)
{
    cloud_.reserve(size);
}

void ReconstructionModule::Visualization::OutputPointCloud(cv::Mat &img, pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    for (int i = cloud_.size() - numNewPts_; i < cloud_.size(); i++)
    {
        cv::Point2f &pt = newPts_[i - cloud_.size() + numNewPts_];
        if (img.channels() == 3)
        {
            cloud_.at(i).r = img.at<cv::Vec3b>(pt.y, pt.x)[2];
            cloud_.at(i).g = img.at<cv::Vec3b>(pt.y, pt.x)[1];
            cloud_.at(i).b = img.at<cv::Vec3b>(pt.y, pt.x)[0];
        }
        else
        {
            cloud_.at(i).r = img.at<unsigned char>(pt.y, pt.x);
            cloud_.at(i).g = img.at<unsigned char>(pt.y, pt.x);
            cloud_.at(i).b = img.at<unsigned char>(pt.y, pt.x);
        }
    }
    numNewPts_ = 0;
    newPts_.clear();
    cloud.clear();
    cloud.reserve(goodIndices_.size());
    for (int inx : goodIndices_)
    {
        cloud.push_back(cloud_.at(inx));
    }
}

}
}
