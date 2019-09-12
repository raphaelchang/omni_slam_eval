#ifndef _TF_UTIL_H_
#define _TF_UTIL_H_

#include <Eigen/Dense>

using namespace Eigen;

namespace omni_slam
{
namespace util
{
namespace TFUtil
{

template <typename T>
inline Matrix<T, 3, 1> CameraFrameToWorldFrame(Matrix<T, 3, 1> cameraFramePt)
{
    Matrix<T, 3, 1> worldFramePt;
    worldFramePt << cameraFramePt(0), cameraFramePt(2), -cameraFramePt(1);
    return worldFramePt;
}

template <typename T>
inline Matrix<T, 3, 1> WorldFrameToCameraFrame(Matrix<T, 3, 1> worldFramePt)
{
    Matrix<T, 3, 1> cameraFramePt;
    cameraFramePt << worldFramePt(0), -worldFramePt(2), worldFramePt(1);
    return cameraFramePt;
}

template <typename T>
inline Matrix<T, 3, 4> InversePoseMatrix(Matrix<T, 3, 4> poseMat)
{
    Matrix<T, 3, 3> rot = poseMat.template block<3, 3>(0, 0);
    Matrix<T, 3, 1> t = poseMat.template block<3, 1>(0, 3);
    Matrix<T, 3, 4> invMat;
    invMat.template block<3, 3>(0, 0) = rot.transpose();
    invMat.template block<3, 1>(0, 3) = -rot.transpose() * t;
    return invMat;
}

template <typename T>
inline Matrix<T, 3, 3> GetRotationFromPoseMatrix(Matrix<T, 3, 4> poseMat)
{
    return poseMat.template block<3, 3>(0, 0);
}

template <typename T>
inline Matrix<T, 3, 1> TransformPoint(Matrix<T, 3, 4> tf, Matrix<T, 3, 1> pt)
{
    Matrix<T, 4, 1> pt_h = pt.homogeneous();
    return tf * pt_h;
}

template <typename T>
inline Matrix<T, 3, 1> RotatePoint(Matrix<T, 3, 3> rot, Matrix<T, 3, 1> pt)
{
    return rot * pt;
}

template <typename T>
inline Matrix<T, 3, 4> QuaternionTranslationToPoseMatrix(Quaternion<T> quat, Matrix<T, 3, 1> translation)
{
    Matrix<T, 3, 4> pose;
    pose.template block<3, 3>(0, 0) = quat.normalized().toRotationMatrix();
    pose.template block<3, 1>(0, 3) = translation;
    return pose;
}

}
}
}

#endif /* _TF_UTIL_H_ */
