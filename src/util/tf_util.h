#ifndef _TF_UTIL_H_
#define _TF_UTIL_H_

#include <Eigen/Dense>

using namespace Eigen;

namespace omni_slam
{
namespace util
{

class TFUtil
{
public:
    static Vector3d CameraFrameToWorldFrame(Vector3d cameraFramePt);
    static Matrix<double, 3, 4> InversePoseMatrix(Matrix<double, 3, 4> poseMat);
    static Vector3d TransformPoint(Matrix<double, 3, 4> tf, Vector3d pt);
};

}
}

#endif /* _TF_UTIL_H_ */
