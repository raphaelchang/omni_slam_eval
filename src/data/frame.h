#ifndef _FRAME_H_
#define _FRAME_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "camera/camera_model.h"

using namespace Eigen;

namespace omni_slam
{
namespace data
{

class Frame
{
public:
    Frame(const int id, cv::Mat &image, cv::Mat &depth_image, Matrix<double, 3, 4>  &pose, double time, camera::CameraModel &camera_model);
    Frame(const int id, cv::Mat &image, cv::Mat &depth_image, double time, camera::CameraModel &camera_model);
    Frame(const int id, cv::Mat &image, Matrix<double, 3, 4>  &pose, double time, camera::CameraModel &camera_model);
    Frame(const int id, cv::Mat &image, double time, camera::CameraModel &camera_model);

    const Matrix<double, 3, 4>& GetPose();
    const Matrix<double, 3, 4>& GetInversePose();
    cv::Mat& GetImage();
    cv::Mat& GetDepthImage();
    const camera::CameraModel& GetCameraModel();
    const int GetID();

    bool HasPose();
    bool HasDepthImage();

    void SetPose(Matrix<double, 3, 4> &pose);
    void SetDepthImage(cv::Mat &depth_image);

    void CompressImages();
    void DecompressImages();
    bool IsCompressed();

private:
    const int id_;
    std::vector<unsigned char> imageComp_;
    std::vector<unsigned char> depthImageComp_;
    cv::Mat image_;
    cv::Mat depthImage_;
    Matrix<double, 3, 4> pose_;
    Matrix<double, 3, 4> invPose_;
    double timeSec_;
    camera::CameraModel &cameraModel_;

    bool hasPose_;
    bool hasDepth_;

    bool isCompressed_{false};
};

}
}
#endif /* _FRAME_H_ */
