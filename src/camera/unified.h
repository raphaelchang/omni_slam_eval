#ifndef _UNIFIED_H_
#define _UNIFIED_H_

#include "camera_model.h"
#include "radtan.h"

namespace omni_slam
{
namespace camera
{

template <typename T = double>
class Unified : public CameraModel<T>
{
    template <typename> friend class Unified;

public:
    Unified(const T fx, const T fy, const T cx, const T cy, const T chi, const T vignette = T(0.))
        : CameraModel<T>("Unified"),
        fx_(fx),
        fy_(fy),
        cx_(cx),
        cy_(cy),
        chi_(chi),
        vignette_(vignette)
    {
        cameraMat_ << fx_, 0., cx_, 0., fy_, cy_, 0., 0., 1.;
        fov_ = GetFOV();
        T theta = M_PI / 2. - fov_ / 2.;
        sinTheta_ = sin(theta);
    }

    Unified(const T fx, const T fy, const T cx, const T cy, const T chi, camera::RadTan<T> *dist, const T vignette = T(0.))
        : Unified(fx, fy, cx, cy, chi, vignette)
    {
        distortionModel_.reset(dist);
    }

    template <typename U>
    Unified(const CameraModel<U> &other)
        : Unified(T(static_cast<const Unified<U>&>(other).fx_), T(static_cast<const Unified<U>&>(other).fy_), T(static_cast<const Unified<U>&>(other).cx_), T(static_cast<const Unified<U>&>(other).cy_), T(static_cast<const Unified<U>&>(other).chi_), new camera::RadTan<T>(*static_cast<const Unified<U>&>(other).distortionModel_), T(static_cast<const Unified<U>&>(other).vignette_))
    {
    }

    bool ProjectToImage(const Matrix<T, 3, 1> &bearing, Matrix<T, 2, 1> &pixel) const
    {
        const T &x = bearing(0);
        const T &y = bearing(1);
        const T &z = bearing(2);

        T d = sqrt(x * x + y * y + z * z);
        T denom = chi_ * d + z;
        pixel(0) = x / denom;
        pixel(1) = y / denom;
        if (distortionModel_)
        {
            Matrix<T, 2, 1> distPixel;
            distortionModel_->Distort(pixel, distPixel);
            pixel(0) = distPixel(0) * fx_ + cx_;
            pixel(1) = distPixel(1) * fy_ + cy_;
        }
        else
        {
            pixel(0) = pixel(0) * fx_ + cx_;
            pixel(1) = pixel(1) * fy_ + cy_;
        }
        if (chi_ > 1.)
        {
            if (z <= sinTheta_ * d)
            {
                return false;
            }
        }
        else
        {
            T w = (chi_ > 1. ? 1. / chi_ : chi_);
            if (z <= -w * d)
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
        T alpha = chi_ / (1. + chi_);
        T mx = (pixel(0) - cx_) / fx_;
        T my = (pixel(1) - cy_) / fy_;
        if (distortionModel_)
        {
            Matrix<T, 2, 1> distPixel;
            distPixel << mx, my;
            Matrix<T, 2, 1> undistPixel;
            distortionModel_->Undistort(distPixel, undistPixel);
            mx = undistPixel(0);
            my = undistPixel(1);
        }

        T r2 = mx * mx + my * my;
        T beta = 1. + (1. - chi_ * chi_) * r2;

        if (beta < 0.)
        {
            return false;
        }

        bearing << mx, my, 1.;

        bearing *= (chi_ + sqrt(beta)) / (1. + r2);
        bearing(2) -= chi_;

        return true;
    }

    T GetFOV() const
    {
        T mx = cx_ / fx_;
        if (chi_ > 1. && vignette_ > 0.)
        {
            mx *= vignette_;
        }
        if (distortionModel_)
        {
            Matrix<T, 2, 1> dist;
            dist << mx, 0;
            Matrix<T, 2, 1> undist;
            distortionModel_->Undistort(dist, undist);
            mx = undist(0);
        }
        T r2;
        T alpha = chi_ / (1. + chi_);
        T beta;
        if (chi_ > 1.)
        {
            if (vignette_ > 0.)
            {
                if (mx * mx < (1. - alpha) * (1. - alpha) / (2. * alpha - 1.))
                {
                    r2 = mx * mx;
                    beta = (chi_ + sqrt(1. + (1. - chi_ * chi_) * r2)) / (1. + r2);
                }
                else
                {
                    r2 = (1. - alpha) * (1. - alpha) / (2. * alpha - 1.);
                    mx = sqrt(r2);
                    beta = chi_ / (1. + r2);
                }
            }
            else
            {
                r2 = (1. - alpha) * (1. - alpha) / (2. * alpha - 1.);
                mx = sqrt(r2);
                beta = chi_ / (1. + r2);
            }
        }
        else
        {
            r2 = mx * mx;
            beta = (chi_ + sqrt(1. + (1. - chi_ * chi_) * r2)) / (1. + r2);
        }
        return 2. * (M_PI / 2 - atan2(beta - chi_, beta * mx));
    }

    typename CameraModel<T>::Type GetType() const
    {
        return CameraModel<T>::kUnified;
    }

private:
    T fx_;
    T fy_;
    T cx_;
    T cy_;
    T chi_;
    T fov_;
    T sinTheta_;
    T vignette_;
    Matrix<T, 3, 3> cameraMat_;
    std::unique_ptr<camera::DistortionModel<T>> distortionModel_;
};

}
}

#endif /* _UNIFIED_H_ */
