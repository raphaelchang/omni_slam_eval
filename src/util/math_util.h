#ifndef _MATH_UTIL_H_
#define _MATH_UTIL_H_

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

}
}
}
#endif /* _MATH_UTIL_H_ */
