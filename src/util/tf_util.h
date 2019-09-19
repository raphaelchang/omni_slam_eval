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

template <typename T>
inline Matrix<T, 3, 4> GetRelativeTransform(Matrix<T, 3, 4> pose1, Matrix<T, 3, 4> pose2)
{
    Matrix<T, 3, 4> rel;
    rel.template block<3, 3>(0, 0) = GetRotationFromPoseMatrix(pose2) * GetRotationFromPoseMatrix(pose1).transpose();
    rel.template block<3, 1>(0, 3) = pose2.template block<3, 1>(0, 3) - rel.template block<3, 3>(0, 0) * pose1.template block<3, 1>(0, 3);
    return rel;
}

template <typename T>
inline Matrix<T, 3, 3> GetEssentialMatrixFromPoses(Matrix<T, 3, 4> pose1, Matrix<T, 3, 4> pose2)
{
    Matrix<T, 3, 4> rel = GetRelativeTransform(pose1, pose2);
    Vector3d t = rel.template block<3, 1>(0, 3);
    Matrix3d tx;
    tx << 0, -t(2), t(1),
       t(2), 0, -t(0),
       -t(1), t(0), 0;
    return tx * GetRotationFromPoseMatrix(rel);
}

}
}
}

#endif /* _TF_UTIL_H_ */
