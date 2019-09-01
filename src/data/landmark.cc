#include "landmark.h"

namespace omni_slam
{
namespace data
{

Landmark::Landmark()
{
}

void Landmark::AddObservation(Feature obs, bool compute_gnd)
{
    if (compute_gnd && obs_.empty())
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

const std::vector<Feature>& Landmark::GetObservations() const
{
    return obs_;
}

std::vector<Feature>& Landmark::GetObservations()
{
    return obs_;
}

bool Landmark::IsObservedInFrame(const int frame_id) const
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

const int Landmark::GetFirstFrameID() const
{
    if (obs_.size() > 0)
    {
        return obs_[0].GetFrame().GetID();
    }
    return -1;
}

const int Landmark::GetNumObservations() const
{
    return obs_.size();
}

const Feature* Landmark::GetObservationByFrameID(const int frame_id) const
{
    if (idToIndex_.find(frame_id) == idToIndex_.end())
    {
        return nullptr;
    }
    return &obs_[idToIndex_.at(frame_id)];
}

Vector3d Landmark::GetGroundTruth() const
{
    return groundTruth_;
}

bool Landmark::HasGroundTruth() const
{
    return hasGroundTruth_;
}

}
}