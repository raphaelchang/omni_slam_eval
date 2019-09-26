#ifndef _DOUBLE_SPHERE_H_
#define _DOUBLE_SPHERE_H_

#include "camera_model.h"

namespace omni_slam
{
namespace camera
{

template <typename T = double>
class DoubleSphere : public CameraModel<T>
{
    template <typename> friend class DoubleSphere;

public:
    DoubleSphere(const T fx, const T fy, const T cx, const T cy, const T chi, const T alpha)
        : CameraModel<T>("DoubleSphere"),
        fx_(fx),
        fy_(fy),
        cx_(cx),
        cy_(cy),
        chi_(chi),
        alpha_(alpha)
    {
        cameraMat_ << fx_, 0., cx_, 0., fy_, cy_, 0., 0., 1.;
        fov_ = GetFOV();
        T theta = M_PI / 2. - fov_ / 2.;
        sinTheta_ = sin(theta);
    }

    template <typename U>
    DoubleSphere(const CameraModel<U> &other)
        : DoubleSphere(T(static_cast<const DoubleSphere<U>&>(other).fx_), T(static_cast<const DoubleSphere<U>&>(other).fy_), T(static_cast<const DoubleSphere<U>&>(other).cx_), T(static_cast<const DoubleSphere<U>&>(other).cy_), T(static_cast<const DoubleSphere<U>&>(other).chi_), T(static_cast<const DoubleSphere<U>&>(other).alpha_))
    {
    }

    bool ProjectToImage(const Matrix<T, 3, 1> &bearing, Matrix<T, 2, 1> &pixel) const
    {
        const T &x = bearing(0);
        const T &y = bearing(1);
        const T &z = bearing(2);

        T d1 = sqrt(x * x + y * y + z * z);
        T d2 = sqrt(x * x + y * y + (chi_ * d1 + z) * (chi_ * d1 + z));
        //T w1 = 0;
        //if (alpha_ <= 0.5 && alpha_ >= 0)
        //{
            //w1 = alpha_ / (1 - alpha_);
        //}
        //else if (alpha_ > 0.5)
        //{
            //w1 = (1 - alpha_) / alpha_;
        //}
        //T w2 = (w1 + chi_) / sqrt(2 * w1 * chi_ + chi_ * chi_ + 1);
        //if (z <= -w2 * d1)
        //{
            //return false;
        //}

        //Matrix<T, 3, 1> bearing_h;
        //bearing_h << x, y, (alpha_ * d2 + (1. - alpha_) * (chi_ * d1 + z));
        //Matrix<T, 3, 1> pixel_h = cameraMat_ * bearing_h;
        //pixel = pixel_h.hnormalized();
        T denom = (alpha_ * d2 + (1. - alpha_) * (chi_ * d1 + z));
        pixel(0) = x * fx_ / denom + cx_;
        pixel(1) = y * fy_ / denom + cy_;
        if (alpha_ > 0.5)
        {
            if (z <= sinTheta_ * d1)
            {
                return false;
            }
        }
        else
        {
            T w1 = alpha_ / (1. - alpha_);
            T w2 = (w1 + chi_) / sqrt(2. * w1 * chi_ + chi_ * chi_ + 1.);
            if (z <= -w2 * d1)
            {
                return false;
            }
        }
        if (pixel(0) > 2. * cx_ || pixel(0) < 0. || pixel(1) > 2. * cy_ || pixel(1) < 0.)
        {
            return false;
        }
        return true;
    }

    bool UnprojectToBearing(const Matrix<T, 2, 1> &pixel, Matrix<T, 3, 1> &bearing) const
    {
        T mx = (pixel(0) - cx_) / fx_;
        T my = (pixel(1) - cy_) / fy_;

        T r2 = mx * mx + my * my;
        T beta1 = 1. - (2. * alpha_ - 1.) * r2;
        if (beta1 < T(0))
        {
            return false;
        }
        T mz = (1. - alpha_ * alpha_ * r2) / (alpha_ * sqrt(beta1) + 1. - alpha_);

        T beta2 = mz * mz + (1. - chi_ * chi_) * r2;

        if (beta2 < 0.)
        {
            return false;
        }

        bearing << mx, my, mz;

        bearing *= (mz * chi_ + sqrt(mz * mz + (1. - chi_ * chi_) * r2)) / (mz * mz + r2);
        bearing(2) -= chi_;

        return true;
    }

    T GetFOV() const
    {
        T mx = cx_ / fx_;
        T r2;
        if (alpha_ > 0.5)
        {
            //r2 = std::min(mx * mx, 1. / (2. * alpha_ - 1.));
            r2 = 1. / (2. * alpha_ - 1.);
        }
        else
        {
            r2 = mx * mx;
        }
        T mz = (1. - alpha_ * alpha_ * r2) / (alpha_ * sqrt(1. - (2. * alpha_ - 1.) * r2) + 1. - alpha_);
        T beta = (mz * chi_ + sqrt(mz * mz + (1. - chi_ * chi_) * r2)) / (mz * mz + r2);
        return 2. * (M_PI / 2 - atan2(beta * mz - chi_, beta * mx));
    }

    typename CameraModel<T>::Type GetType() const
    {
        return CameraModel<T>::kDoubleSphere;
    }

private:
    T fx_;
    T fy_;
    T cx_;
    T cy_;
    T chi_;
    T alpha_;
    T fov_;
    T sinTheta_;
    Matrix<T, 3, 3> cameraMat_;
};

}
}

#endif /* _DOUBLE_SPHERE_  */
