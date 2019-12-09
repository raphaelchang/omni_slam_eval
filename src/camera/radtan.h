#ifndef _RADTAN_H_
#define _RADTAN_H_

#include "distortion_model.h"

#define DIST_INV_TOLERANCE 1e-8

namespace omni_slam
{
namespace camera
{

template <typename T = double>
class RadTan : public DistortionModel<T>
{
    template <typename> friend class RadTan;

public:
    RadTan(const T k1, const T k2, const T p1, const T p2)
        : DistortionModel<T>("RadTan"),
        k1_(k1),
        k2_(k2),
        p1_(p1),
        p2_(p2)
    {
    }

    template <typename U>
    RadTan(const DistortionModel<U> &other)
        : RadTan(T(static_cast<const RadTan<U>&>(other).k1_), T(static_cast<const RadTan<U>&>(other).k2_), T(static_cast<const RadTan<U>&>(other).p1_), T(static_cast<const RadTan<U>&>(other).p2_))
    {
    }

    bool Undistort(const Matrix<T, 2, 1> &pixel_dist, Matrix<T, 2, 1> &pixel_undist) const
    {
        Matrix<T, 2, 1> pixelIter = pixel_dist;
        const int n = 5;

        for (int i = 0; i < n; i++)
        {
            Matrix<T, 2, 1> pixTemp = pixelIter;

            const T &x = pixTemp(0);
            const T &y = pixTemp(1);

            const T r2 = x * x + y * y;
            const T r4 = r2 * r2;
            const T kr = 1. + k1_ * r2 + k2_ * r4;
            const T deltaX = 2. * p1_ * x * y + p2_ * (r2 + 2. * x * x);
            const T deltaY = 2. * p2_ * x * y + p1_ * (r2 + 2. * y * y);
            pixelIter(0) = (pixel_dist(0) - deltaX) / kr;
            pixelIter(1) = (pixel_dist(1) - deltaY) / kr;

            //pixTemp(0) += x * kr + 2. * p1_ * x * y + p2_ * (r2 + 2. * x * x);
            //pixTemp(1) += y * kr + 2. * p2_ * x * y + p1_ * (r2 + 2. * y * y);
            //const T duf_du = 1. + kr + 2. * k1_ * x * x + 4. * k2_ * r2 * x * x + 2. * p1_ * y + 6. * p2_ * x;
            //const T dvf_dv = 1. + kr + 2. * k1_ * y * y + 4. * k2_ * r2 * y * y + 2. * p2_ * x + 6. * p1_ * y;
            //const T duf_dv = 2. * k1_ * x * y + 4. * k2_ * r2 * x * y + 2. * p1_ * x + 2. * p2_ * y;
            //const T dvf_du = duf_dv;
            //Matrix<T, 2, 2> F;
            //F << duf_du, duf_dv, dvf_du, dvf_dv;
            //Matrix<T, 2, 1> e(pixel_dist - pixTemp);
            //Matrix<T, 2, 1> du = (F.transpose() * F).inverse() * F.transpose() * e;
            //pixelIter += du;
            //if (e.dot(e) <= DIST_INV_TOLERANCE)
            //{
                //break;
            //}
        }
        pixel_undist = pixelIter;
    }

    bool Distort(const Matrix<T, 2, 1> &pixel_undist, Matrix<T, 2, 1> &pixel_dist) const
    {
        const T &x = pixel_undist(0);
        const T &y = pixel_undist(1);

        const T r2 = x * x + y * y;
        const T r4 = r2 * r2;
        const T kr = 1.0 + k1_ * r2 + k2_ * r4;

        pixel_dist(0) = x * kr + 2. * p1_ * x * y + p2_ * (r2 + 2. * x * x);
        pixel_dist(1) = y * kr + 2. * p2_ * x * y + p1_ * (r2 + 2. * y * y);
    }

private:
    T k1_;
    T k2_;
    T p1_;
    T p2_;
};

}
}
#endif /* _RADTAN_H_ */
