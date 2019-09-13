#ifndef _ODOMETRY_MODULE_H_
#define _ODOMETRY_MODULE_H_

#include <vector>
#include <set>
#include <memory>

#include "odometry/pnp.h"
#include "data/landmark.h"

#include <pcl/common/projection_matrix.h>
#include <pcl/point_types.h>

using namespace Eigen;

namespace omni_slam
{
namespace module
{

class OdometryModule
{
public:
    struct Stats
    {
    };

    OdometryModule(std::unique_ptr<odometry::PNP> &pnp);
    OdometryModule(std::unique_ptr<odometry::PNP> &&pnp);

    void Update(std::vector<data::Landmark> &landmarks, data::Frame &frame);

    Stats& GetStats();

private:
    std::shared_ptr<odometry::PNP> pnp_;

    Stats stats_;
};

}
}

#endif /* _ODOMETRY_MODULE_H_ */
