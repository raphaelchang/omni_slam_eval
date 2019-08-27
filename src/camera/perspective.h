#ifndef _PERSPECTIVE_H_
#define _PERSPECTIVE_H_

#include "camera_model.h"

namespace omni_slam
{
namespace camera
{

class Perspective : public CameraModel
{
public:
    Perspective(const double fx, const double fy, const double cx, const double cy);

    bool ProjectToImage(const Vector3d &bearing, Vector2d &pixel) const;
    bool UnprojectToBearing(const Vector2d &pixel, Vector3d &bearing) const;
    double GetFOV();

private:
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    Matrix3d cameraMat_;
};

}
}

#endif /* _PERSPECTIVE_H_ */
