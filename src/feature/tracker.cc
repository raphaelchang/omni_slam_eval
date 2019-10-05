#include "tracker.h"

namespace omni_slam
{
namespace feature
{

Tracker::Tracker(const int keyframe_interval)
    : keyframeInterval_(keyframe_interval),
    prevFrame_(nullptr)
{
}

void Tracker::Init(data::Frame &init_frame)
{
    frameNum_ = 0;
    prevId_ = init_frame.GetID();
    prevFrame_ = &init_frame;
    keyframeId_ = init_frame.GetID();
    keyframeImg_ = init_frame.GetImage().clone();
    if (init_frame.HasStereoImage())
    {
        keyframeStereoImg_ = init_frame.GetStereoImage().clone();
    }
}

const data::Frame* Tracker::GetLastKeyframe()
{
    return prevFrame_;
}

int Tracker::Track(std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, std::vector<double> &errors, bool stereo)
{
    if (keyframeImg_.empty())
    {
        return 0;
    }
    bool wasCompressed = cur_frame.IsCompressed();

    int count = DoTrack(landmarks, cur_frame, errors, stereo);

    prevId_ = cur_frame.GetID();
    prevFrame_ = &cur_frame;
    if (++frameNum_ % keyframeInterval_ == 0)
    {
        keyframeId_ = cur_frame.GetID();
        keyframeImg_ = cur_frame.GetImage().clone();
        if (cur_frame.HasStereoImage())
        {
            keyframeStereoImg_ = cur_frame.GetStereoImage().clone();
        }
    }
    if (wasCompressed && !cur_frame.IsCompressed())
    {
        cur_frame.CompressImages();
    }

    return count;
}

}
}
