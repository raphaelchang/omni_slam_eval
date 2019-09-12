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
    const std::vector<Feature>& GetObservationsForEstimate() const;
    bool IsObservedInFrame(const int frame_id) const;
    const int GetFirstFrameID() const;
    const int GetNumObservations() const;
    const Feature* GetObservationByFrameID(const int frame_id) const;
    void SetEstimatedPosition(const Vector3d &pos);

    Vector3d GetGroundTruth() const;
    Vector3d GetEstimatedPosition() const;
    bool HasGroundTruth() const;
    bool HasEstimatedPosition() const;

private:
    std::vector<Feature> obs_;
    std::vector<Feature> obsForEst_;
    std::map<int, int> idToIndex_;
    Vector3d groundTruth_;
    Vector3d posEstimate_;
    bool hasGroundTruth_{false};
    bool hasPosEstimate_{false};
};

}
}

#endif /* _LANDMARK_H_ */
