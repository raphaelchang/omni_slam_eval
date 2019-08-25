#include "landmark.h"

namespace omni_slam
{
namespace data
{

Landmark::Landmark()
{
}

void Landmark::AddObservation(Feature obs)
{
    if (obs_.empty())
    {
        if (obs.HasWorldPoint())
        {
            groundTruth_ = obs.GetWorldPoint();
            hasGroundTruth_ = true;
        }
    }
    if (idToIndex_.find(obs.GetFrame().GetID()) != idToIndex_.end())
    {
        return;
    }
    idToIndex_[obs.GetFrame().GetID()] = obs_.size();
    obs_.push_back(obs);
}

std::vector<Feature>& Landmark::GetObservations()
{
    return obs_;
}

bool Landmark::IsObservedInFrame(const int frame_id)
{
    for (Feature f : obs_)
    {
        if (f.GetFrame().GetID() == frame_id)
        {
            return true;
        }
    }
    return false;
}

const int Landmark::GetFirstFrameID()
{
    if (obs_.size() > 0)
    {
        return obs_[0].GetFrame().GetID();
    }
    return -1;
}

const int Landmark::GetNumObservations()
{
    return obs_.size();
}

Feature* Landmark::GetObservationByFrameID(const int frame_id)
{
    if (idToIndex_.find(frame_id) == idToIndex_.end())
    {
        return nullptr;
    }
    return &obs_[idToIndex_[frame_id]];
}

Vector3d Landmark::GetGroundTruth()
{
    return groundTruth_;
}

bool Landmark::HasGroundTruth()
{
    return hasGroundTruth_;
}

}
}
