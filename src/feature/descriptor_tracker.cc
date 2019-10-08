#include "descriptor_tracker.h"
#include "util/tf_util.h"

namespace omni_slam
{
namespace feature
{

DescriptorTracker::DescriptorTracker(std::string detector_type, std::string descriptor_type, std::map<std::string, double> det_args, std::map<std::string, double> desc_args, const float match_thresh, const int keyframe_interval)
    : Tracker(keyframe_interval),
    Matcher(descriptor_type, match_thresh),
    Detector(detector_type, descriptor_type, det_args, desc_args)
{
}

int DescriptorTracker::DoTrack(std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, std::vector<double> &errors, bool stereo)
{
    std::vector<data::Landmark> curLandmarks;
    std::vector<data::Landmark> prevLandmarks;
    int imsize = std::max(cur_frame.GetImage().rows, cur_frame.GetImage().cols);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Region::rs.size() - 1; i++)
    {
        for (int j = 0; j < Region::ts.size() - 1; j++)
        {
            feature::Detector detector(*static_cast<Detector*>(this));
            detector.DetectInRadialRegion(cur_frame, curLandmarks, feature::Region::rs[i] * imsize, feature::Region::rs[i+1] * imsize, feature::Region::ts[j], feature::Region::ts[j+1]);
            detector.DetectInRadialRegion(cur_frame, curLandmarks, feature::Region::rs[i] * imsize, feature::Region::rs[i+1] * imsize, feature::Region::ts[j], feature::Region::ts[j+1], true);
        }
    }
    std::vector<int> origInx;
    int i = 0;
    for (const data::Landmark &landmark : landmarks)
    {
        if (landmark.IsObservedInFrame(keyframeId_))
        {
            data::Landmark l;
            l.AddObservation(*landmark.GetObservationByFrameID(keyframeId_), false);
            if (stereo)
            {
                const data::Feature* stereoFeat = landmark.GetStereoObservationByFrameID(keyframeId_);
                if (stereoFeat != nullptr)
                {
                    l.AddStereoObservation(*stereoFeat);
                }
            }
            prevLandmarks.push_back(l);
            origInx.push_back(i);
        }
        i++;
    }
    std::vector<data::Landmark> matches;
    std::vector<data::Landmark> stereoMatches;
    std::vector<int> indices;
    std::vector<std::vector<double>> distances;
    std::vector<std::vector<double>> stereoDistances;
    std::vector<int> stereoIndices;
    Match(curLandmarks, prevLandmarks, matches, distances, indices, false);
    if (stereo)
    {
        Match(curLandmarks, prevLandmarks, stereoMatches, stereoDistances, stereoIndices, true);
    }
    errors.clear();
    int numGood = 0;
    for (int i = 0; i < matches.size(); i++)
    {
        data::Landmark &landmark = landmarks[origInx[indices[i]]];
        data::Feature feat(cur_frame, matches[i].GetObservationByFrameID(cur_frame.GetID())->GetKeypoint(), matches[i].GetObservationByFrameID(cur_frame.GetID())->GetDescriptor());
        landmark.AddObservation(feat);
        errors.push_back(distances[i][0]);
        numGood++;
    }
    for (int i = 0; i < stereoMatches.size(); i++)
    {
        data::Landmark &landmark = landmarks[origInx[stereoIndices[i]]];
        if (!landmark.IsObservedInFrame(cur_frame.GetID()))
        {
            continue;
        }
        data::Feature feat(cur_frame, stereoMatches[i].GetObservationByFrameID(cur_frame.GetID())->GetKeypoint(), stereoMatches[i].GetObservationByFrameID(cur_frame.GetID())->GetDescriptor(), true);
        landmark.AddStereoObservation(feat);
    }

    return numGood;
}

}
}
