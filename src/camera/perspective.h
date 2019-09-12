#ifndef _PERSPECTIVE_H_
#define _PERSPECTIVE_H_

#include "camera_model.h"

namespace omni_slam
{
namespace camera
{

template <typename T = double>
class Perspective : public CameraModel<T>
{
    template <typename> friend class Perspective;

public:
    Perspective(const T fx, const T fy, const T cx, const T cy)
        : CameraModel<T>("Perspective"),
        fx_(fx),
        fy_(fy),
        cx_(cx),
        cy_(cy)
    {
        cameraMat_ << fx_, 0., cx_, 0., fy_, cy_, 0., 0., 1.;
    }

    template <typename U>
    Perspective(const CameraModel<U> &other)
        : Perspective(T(static_cast<const Perspective<U>&>(other).fx_), T(static_cast<const Perspective<U>&>(other).fy_), T(static_cast<const Perspective<U>&>(other).cx_), T(static_cast<const Perspective<U>&>(other).cy_))
    {
    }

    bool ProjectToImage(const Matrix<T, 3, 1> &bearing, Matrix<T, 2, 1> &pixel) const
    {
        const T &x = bearing(0);
        const T &y = bearing(1);
        const T &z = bearing(2);
        //Matrix<T, 3, 1> pixel_h = cameraMat_ * bearing;
        //pixel = pixel_h.hnormalized();
        pixel(0) = x * fx_ / z + cx_;
        pixel(1) = y * fy_ / z + cy_;
        if (pixel(0) > 2. * cx_ || pixel(0) < 0. || pixel(1) > 2. * cy_ || pixel(1) < 0.)
        {
            return false;
        }
        return true;
    }

    bool UnprojectToBearing(const Matrix<T, 2, 1> &pixel, Matrix<T, 3, 1> &bearing) const
    {
        Matrix<T, 3, 1> pixel_h;
        T mx = (pixel(0) - cx_) / fx_;
        T my = (pixel(1) - cy_) / fy_;
        pixel_h << mx, my, 1;
        bearing = 1. / sqrt(mx * mx + my * my + 1.) * pixel_h;
        return true;
    }

    T GetFOV() const
    {
        return 2. * atan(cx_ / fx_);
    }

    typename CameraModel<T>::Type GetType() const
    {
        return CameraModel<T>::kPerspective;
    }

private:
    T fx_;
    T fy_;
    T cx_;
    T cy_;
    Matrix<T, 3, 3> cameraMat_;
};

}
}

#endif /* _PERSPECTIVE_H_ */
