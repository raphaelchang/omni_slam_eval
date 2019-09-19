#include "landmark.h"

namespace omni_slam
{
namespace data
{

int Landmark::lastLandmarkId_ = 0;

Landmark::Landmark()
    : id_(lastLandmarkId_++)
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
        if (!obs.GetFrame().HasStereoImage() && obs.HasEstimatedWorldPoint())
        {
            posEstimate_ = obs.GetEstimatedWorldPoint();
            estFrameIds_.insert(obs.GetFrame().GetID());
            hasPosEstimate_ = true;
        }
    }
    if (idToIndex_.find(obs.GetFrame().GetID()) != idToIndex_.end())
    {
        return;
    }
    idToIndex_[obs.GetFrame().GetID()] = obs_.size();
    obs_.push_back(obs);
}

void Landmark::AddStereoObservation(Feature obs)
{
    if (idToStereoIndex_.find(obs.GetFrame().GetID()) != idToStereoIndex_.end())
    {
        return;
    }
    idToStereoIndex_[obs.GetFrame().GetID()] = stereoObs_.size();
    stereoObs_.push_back(obs);
}

const std::vector<Feature>& Landmark::GetObservations() const
{
    return obs_;
}

std::vector<Feature>& Landmark::GetObservations()
{
    return obs_;
}

const std::vector<Feature>& Landmark::GetStereoObservations() const
{
    return stereoObs_;
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

const Feature* Landmark::GetStereoObservationByFrameID(const int frame_id) const
{
    if (idToStereoIndex_.find(frame_id) == idToStereoIndex_.end())
    {
        return nullptr;
    }
    return &stereoObs_[idToStereoIndex_.at(frame_id)];
}

void Landmark::SetEstimatedPosition(const Vector3d &pos, const std::vector<int> &frame_ids)
{
    posEstimate_ = pos;
    estFrameIds_ = std::unordered_set<int>(frame_ids.begin(), frame_ids.end());
    hasPosEstimate_ = true;
}

void Landmark::SetEstimatedPosition(const Vector3d &pos)
{
    posEstimate_ = pos;
    hasPosEstimate_ = true;
}

const int Landmark::GetID() const
{
    return id_;
}

Vector3d Landmark::GetGroundTruth() const
{
    return groundTruth_;
}

Vector3d Landmark::GetEstimatedPosition() const
{
    return posEstimate_;
}

bool Landmark::HasGroundTruth() const
{
    return hasGroundTruth_;
}

bool Landmark::HasEstimatedPosition() const
{
    return hasPosEstimate_;
}

bool Landmark::IsEstimatedByFrame(const int frame_id) const
{
    return estFrameIds_.find(frame_id) != estFrameIds_.end();
}

}
}
