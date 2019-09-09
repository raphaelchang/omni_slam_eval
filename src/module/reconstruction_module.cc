#include "reconstruction_module.h"

using namespace std;

namespace omni_slam
{
namespace module
{

ReconstructionModule::ReconstructionModule(std::unique_ptr<reconstruction::Triangulator> &triangulator)
    : triangulator_(std::move(triangulator))
{
}

ReconstructionModule::ReconstructionModule(std::unique_ptr<reconstruction::Triangulator> &&triangulator)
    : ReconstructionModule(triangulator)
{
}

void ReconstructionModule::Update(std::vector<data::Landmark> &landmarks)
{
    triangulator_->Triangulate(landmarks);

    visualization_.Reserve(landmarks.size());
    for (int i = 0; i < lastLandmarksSize_; i++)
    {
        if (landmarks[i].HasEstimatedPosition())
        {
            visualization_.UpdatePoint(i, landmarks[i].GetEstimatedPosition());
        }
    }
    for (int i = lastLandmarksSize_; i < landmarks.size(); i++)
    {
        cv::Point2f pt = landmarks[i].GetObservations()[0].GetKeypoint().pt;
        visualization_.AddPoint(pt);
    }
    lastLandmarksSize_ = landmarks.size();
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
