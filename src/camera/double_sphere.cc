#include "double_sphere.h"
#include <cmath>
#include <algorithm>

namespace omni_slam
{
namespace camera
{

DoubleSphere::DoubleSphere(const double fx, const double fy, const double cx, const double cy, const double chi, const double alpha)
    : CameraModel("DoubleSphere"),
    fx_(fx),
    fy_(fy),
    cx_(cx),
    cy_(cy),
    chi_(chi),
    alpha_(alpha)
{
    cameraMat_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
    fov_ = GetFOV();
}

bool DoubleSphere::ProjectToImage(const Vector3d &bearing, Vector2d &pixel) const
{
    double x = bearing(0);
    double y = bearing(1);
    double z = bearing(2);

    double d1 = bearing.norm();
    double d2 = sqrt(x * x + y * y + (chi_ * d1 + z) * (chi_ * d1 + z));
    double w1 = 0;
    if (alpha_ <= 0.5 && alpha_ > 0)
    {
        w1 = alpha_ / (1 - alpha_);
    }
    else if (alpha_ > 0.5)
    {
        w1 = (1 - alpha_) / alpha_;
    }
    //double w2 = (w1 + chi_) / sqrt(2 * w1 * chi_ + chi_ * chi_ + 1);
    //if (z <= -w2 * d1)
    //{
    //    return false;
    //}

    Vector3d bearing_h;
    bearing_h << x, y, (alpha_ * d2 + (1 - alpha_) * (chi_ * d1 + z));
    Vector3d pixel_h = cameraMat_ * bearing_h;
    pixel = pixel_h.hnormalized();
    if (pixel(0) > 2 * cx_ || pixel(0) < 0 || pixel(1) > 2 * cy_ || pixel(1) < 0)
    {
        return false;
    }
    if (alpha_ > 0.5)
    {
        double theta = M_PI / 2 - fov_ / 2;
        if (z <= sin(theta) * d1)
        {
            return false;
        }
    }
    return true;
}

bool DoubleSphere::UnprojectToBearing(const Vector2d &pixel, Vector3d &bearing) const
{
    double mx = (pixel(0) - cx_) / fx_;
    double my = (pixel(1) - cy_) / fy_;

    double r2 = mx * mx + my * my;
    double beta1 = 1. - (2. * alpha_ - 1) * r2;
    if (beta1 < 0)
    {
        return false;
    }
    double mz = (1. - alpha_ * alpha_ * r2) / (alpha_ * sqrt(beta1) + 1. - alpha_);

    double beta2 = mz * mz + (1. - chi_ * chi_) * r2;

    if (beta2 < 0)
    {
        return false;
    }

    bearing << mx, my, mz;

    bearing *= (mz * chi_ + sqrt(mz * mz + (1. - chi_ * chi_) * r2)) / (mz * mz + r2);
    bearing(2) -= chi_;

    return true;
}

double DoubleSphere::GetFOV()
{
    double mx = cx_ / fx_;
    double r2;
    if (alpha_ > 0.5)
    {
        r2 = std::min(mx * mx, 1 / (2 * alpha_ - 1));
    }
    else
    {
        r2 = mx * mx;
    }
    double mz = (1 - alpha_ * alpha_ * r2) / (alpha_ * sqrt(1 - (2 * alpha_ - 1) * r2) + 1 - alpha_);
    double beta = (mz * chi_ + sqrt(mz * mz + (1 - chi_ * chi_) * r2)) / (mz * mz + r2);
    return 2 * (M_PI / 2 - atan2(beta * mz - chi_, beta * mx));
}

}
}
