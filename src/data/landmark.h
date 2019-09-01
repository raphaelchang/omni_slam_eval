#ifndef _LANDMARK_H_
#define _LANDMARK_H_

#include "feature.h"
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

namespace omni_slam
{
namespace data
{

class Landmark
{
public:
    Landmark();
    void AddObservation(Feature obs, bool compute_gnd = true);
    const std::vector<Feature>& GetObservations() const;
    std::vector<Feature>& GetObservations();
    bool IsObservedInFrame(const int frame_id) const;
    const int GetFirstFrameID() const;
    const int GetNumObservations() const;
    const Feature* GetObservationByFrameID(const int frame_id) const;

    Vector3d GetGroundTruth() const;
    bool HasGroundTruth() const;
private:
    std::vector<Feature> obs_;
    std::map<int, int> idToIndex_;
    Vector3d groundTruth_;
    bool hasGroundTruth_;
};

}
}

#endif /* _LANDMARK_H_ */
