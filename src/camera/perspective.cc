#include "perspective.h"
#include <cmath>

namespace omni_slam
{
namespace camera
{

Perspective::Perspective(const double fx, const double fy, const double cx, const double cy)
    : CameraModel("Perspective"),
    fx_(fx),
    fy_(fy),
    cx_(cx),
    cy_(cy)
{
    cameraMat_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
}

bool Perspective::ProjectToImage(const Vector3d &bearing, Vector2d &pixel)
{
    Vector3d pixel_h = cameraMat_ * bearing;
    pixel = pixel_h.hnormalized();
    if (pixel(0) > 2 * cx_ || pixel(0) < 0 || pixel(1) > 2 * cy_ || pixel(1) < 0)
    {
        return false;
    }
    return true;
}

bool Perspective::UnprojectToBearing(const Vector2d &pixel, Vector3d &bearing)
{
    Vector3d pixel_h;
    double mx = (pixel(0) - cx_) / fx_;
    double my = (pixel(1) - cy_) / fy_;
    pixel_h << mx, my, 1;
    bearing = 1. / sqrt(mx * mx + my * my + 1) * pixel_h;
    return true;
}

double Perspective::GetFOV()
{
    return 2 * atan(cx_ / fx_);
}

}
}
