#include "tf_util.h"

namespace omni_slam
{
namespace util
{

Vector3d TFUtil::CameraFrameToWorldFrame(Vector3d cameraFramePt)
{
    Vector3d worldFramePt;
    worldFramePt << cameraFramePt(0), cameraFramePt(2), -cameraFramePt(1);
    return worldFramePt;
}

Vector3d TFUtil::WorldFrameToCameraFrame(Vector3d worldFramePt)
{
    Vector3d cameraFramePt;
    cameraFramePt << worldFramePt(0), -worldFramePt(2), worldFramePt(1);
    return cameraFramePt;
}

Matrix<double, 3, 4> TFUtil::InversePoseMatrix(Matrix<double, 3, 4> poseMat)
{
    Matrix3d rot = poseMat.block<3, 3>(0, 0);
    Vector3d t = poseMat.block<3, 1>(0, 3);
    Matrix<double, 3, 4> invMat;
    invMat.block<3, 3>(0, 0) = rot.transpose();
    invMat.block<3, 1>(0, 3) = -rot.transpose() * t;
    return invMat;
}

Matrix3d TFUtil::GetRotationFromPoseMatrix(Matrix<double, 3, 4> poseMat)
{
    return poseMat.block<3, 3>(0, 0);
}

Vector3d TFUtil::TransformPoint(Matrix<double, 3, 4> tf, Vector3d pt)
{
    Vector4d pt_h = pt.homogeneous();
    return tf * pt_h;
}

Vector3d TFUtil::RotatePoint(Matrix3d rot, Vector3d pt)
{
    return rot * pt;
}

Matrix<double, 3, 4> TFUtil::QuaternionTranslationToPoseMatrix(Quaterniond quat, Vector3d translation)
{
    Matrix<double, 3, 4> pose;
    pose.block<3, 3>(0, 0) = quat.normalized().toRotationMatrix();
    pose.block<3, 1>(0, 3) = translation;
    return pose;
}

}
}
