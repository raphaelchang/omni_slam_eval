#ifndef _TRIANGULATOR_H_
#define _TRIANGULATOR_H_

#include "data/landmark.h"
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

namespace omni_slam
{
namespace reconstruction
{

class Triangulator
{
public:
    Triangulator(double max_reprojection_error = 0.0, double min_triangulation_angle = 1.0);

    int Triangulate(std::vector<data::Landmark> &landmarks) const;

private:
    bool TriangulateNViews(const std::vector<data::Feature> &views, Vector3d &point) const;
    bool CheckReprojectionErrors(const std::vector<data::Feature> &views, const Vector3d &point) const;
    bool CheckAngleCoverage(const std::vector<Vector3d> &bearings) const;

    double cosMinTriangulationAngle_;
    double maxReprojError_;
};

}
}

#endif /* _TRIANGULATOR_H_ */
