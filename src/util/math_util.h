#ifndef _MATH_UTIL_H_
#define _MATH_UTIL_H_

#include <Eigen/Dense>
using namespace Eigen;

namespace omni_slam
{
namespace util
{
namespace MathUtil
{

template <typename T>
inline T FastAtan2(T y, T x)
{
    const T ONEQTR_PI = M_PI / 4.0;
    const T THRQTR_PI = 3.0 * M_PI / 4.0;
    T r, angle;
    T abs_y = std::fabs(y) + 1e-10f;
    if (x < 0.0f)
    {
        r = (x + abs_y) / (abs_y - x);
        angle = THRQTR_PI;
    }
    else
    {
        r = (x - abs_y) / (x + abs_y);
        angle = ONEQTR_PI;
    }
    angle += (0.1963f * r * r - 0.9817f) * r;
    if (y < 0.0f)
    {
        return -angle;
    }
    else
    {
        return angle;
    }
}

template<typename T>
inline bool Roots(T b, T c, T &r1, T &r2)
{
    T v = b * b - 4.0 * c;
    if (v < 0.)
    {
        r1 = r2 = 0.5 * b;
        return false;
    }

    T y = std::sqrt(v);

    if (b < 0)
    {
        r1 = 0.5 * (-b + y);
        r2 = 0.5 * (-b - y);
    }
    else
    {
        r1 = 2.0 * c / (-b + y);
        r2 = 2.0 * c / (-b - y);
    }
    return true;
}

template<typename T>
inline Matrix<T, 1, 10> MultiplyDegOnePoly(const Matrix<T, 1, 4> &a, const Matrix<T, 1, 4> &b)
{
    Matrix<T, 1, 10> output;
    output(0) = a(0) * b(0);
    output(1) = a(0) * b(1) + a(1) * b(0);
    output(2) = a(1) * b(1);
    output(3) = a(0) * b(2) + a(2) * b(0);
    output(4) = a(1) * b(2) + a(2) * b(1);
    output(5) = a(2) * b(2);
    output(6) = a(0) * b(3) + a(3) * b(0);
    output(7) = a(1) * b(3) + a(3) * b(1);
    output(8) = a(2) * b(3) + a(3) * b(2);
    output(9) = a(3) * b(3);
    return output;
}

template<typename T>
inline Matrix<T, 1, 20> MultiplyDegTwoDegOnePoly(const Matrix<T, 1, 10> &a, const Matrix<T, 1, 4> &b)
{
    Matrix<T, 1, 20> output;
    output(0) = a(0) * b(0);
    output(1) = a(0) * b(1) + a(1) * b(0);
    output(2) = a(1) * b(1) + a(2) * b(0);
    output(3) = a(2) * b(1);
    output(4) = a(0) * b(2) + a(3) * b(0);
    output(5) = a(1) * b(2) + a(3) * b(1) + a(4) * b(0);
    output(6) = a(2) * b(2) + a(4) * b(1);
    output(7) = a(3) * b(2) + a(5) * b(0);
    output(8) = a(4) * b(2) + a(5) * b(1);
    output(9) = a(5) * b(2);
    output(10) = a(0) * b(3) + a(6) * b(0);
    output(11) = a(1) * b(3) + a(6) * b(1) + a(7) * b(0);
    output(12) = a(2) * b(3) + a(7) * b(1);
    output(13) = a(3) * b(3) + a(6) * b(2) + a(8) * b(0);
    output(14) = a(4) * b(3) + a(7) * b(2) + a(8) * b(1);
    output(15) = a(5) * b(3) + a(8) * b(2);
    output(16) = a(6) * b(3) + a(9) * b(0);
    output(17) = a(7) * b(3) + a(9) * b(1);
    output(18) = a(8) * b(3) + a(9) * b(2);
    output(19) = a(9) * b(3);
    return output;
}

}
}
}
#endif /* _MATH_UTIL_H_ */
