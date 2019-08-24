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
    Frame(const int id, cv::Mat &image, cv::Mat &depth_image, Matrix<double, 3, 4>  &pose, camera::CameraModel &camera_model);
    Frame(const int id, cv::Mat &image, cv::Mat &depth_image, camera::CameraModel &camera_model);
    Frame(const int id, cv::Mat &image, Matrix<double, 3, 4>  &pose, camera::CameraModel &camera_model);
    Frame(const int id, cv::Mat &image, camera::CameraModel &camera_model);

    const Matrix<double, 3, 4>& GetPose();
    const cv::Mat& GetImage();
    const cv::Mat& GetDepthImage();
    const camera::CameraModel& GetCameraModel();
    const int GetID();

    bool HasPose();
    bool HasDepthImage();

    void SetPose(Matrix<double, 3, 4> &pose);
    void SetDepthImage(cv::Mat &depth_image);
private:
    const int id_;
    const cv::Mat image_;
    cv::Mat depthImage_;
    Matrix<double, 3, 4> pose_;
    camera::CameraModel &cameraModel_;

    bool hasPose_;
    bool hasDepth_;
};

}
}
#endif /* _FRAME_H_ */
