#include "math_util.h"
#include <math.h>

namespace omni_slam
{
namespace util
{

double MathUtil::FastAtan2(double y, double x)
{
    const float ONEQTR_PI = M_PI / 4.0;
    const float THRQTR_PI = 3.0 * M_PI / 4.0;
    float r, angle;
    float abs_y = fabs(y) + 1e-10f;
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

}
}
