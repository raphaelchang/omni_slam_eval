#ifndef _DOUBLE_SPHERE_H_
#define _DOUBLE_SPHERE_H_

#include "camera_model.h"

namespace omni_slam
{
namespace camera
{

class DoubleSphere : public CameraModel
{
public:
    DoubleSphere(const double fx, const double fy, const double cx, const double cy, const double chi, const double alpha);

    bool ProjectToImage(const Vector3d &bearing, Vector2d &pixel);
    bool UnprojectToBearing(const Vector2d &pixel, Vector3d &bearing);
    double GetFOV();

private:
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    double chi_;
    double alpha_;
    double fov_;
    Matrix3d cameraMat_;
};

}
}

#endif /* _DOUBLE_SPHERE_  */
