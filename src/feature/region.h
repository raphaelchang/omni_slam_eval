#ifndef _REGION_H_
#define _REGION_H_

namespace omni_slam
{
namespace feature
{
namespace Region
{

const std::vector<double> rs{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7};
const std::vector<double> ts{-M_PI, -3 * M_PI / 4, -M_PI / 2, -M_PI / 4, 0, M_PI / 4, M_PI / 2, 3 * M_PI / 4, M_PI};

}
}
}

#endif /* _REGION_H_ */
