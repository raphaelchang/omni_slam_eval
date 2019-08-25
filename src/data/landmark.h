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
    void AddObservation(Feature obs);
    std::vector<Feature>& GetObservations();
    bool IsObservedInFrame(const int frame_id);
    const int GetFirstFrameID();
    const int GetNumObservations();
    Feature* GetObservationByFrameID(const int frame_id);

    Vector3d GetGroundTruth();
    bool HasGroundTruth();
private:
    std::vector<Feature> obs_;
    std::map<int, int> idToIndex_;
    Vector3d groundTruth_;
    bool hasGroundTruth_;
};

}
}

#endif /* _LANDMARK_H_ */
