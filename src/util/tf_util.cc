#include "tf_util.h"

namespace omni_slam
{
namespace util
{

Vector3d TFUtil::CameraFrameToWorldFrame(Vector3d cameraFramePt)
{
    Vector3d worldFramePt;
    worldFramePt << cameraFramePt(0), -cameraFramePt(2), cameraFramePt(1);
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

Vector3d TFUtil::TransformPoint(Matrix<double, 3, 4> tf, Vector3d pt)
{
    Vector4d pt_h;
    pt_h << pt(0), pt(1), pt(2), 1;
    return tf * pt_h;
}

}
}
