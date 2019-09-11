#ifndef _MATH_UTIL_H_
#define _MATH_UTIL_H_

namespace omni_slam
{
namespace util
{

class MathUtil
{
public:
    template <typename T>
    static T FastAtan2(T y, T x)
    {
        const T ONEQTR_PI = M_PI / 4.0;
        const T THRQTR_PI = 3.0 * M_PI / 4.0;
        T r, angle;
        T abs_y = fabs(y) + 1e-10f;
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
};

}
}
#endif /* _MATH_UTIL_H_ */
