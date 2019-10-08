#ifndef _TRACKER_H_
#define _TRACKER_H_

#include "data/frame.h"
#include "data/landmark.h"

namespace omni_slam
{
namespace feature
{

class Tracker
{
public:
    Tracker(const int keyframe_interval = 1);

    virtual void Init(data::Frame &init_frame);
    int Track(std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, std::vector<double> &errors, bool stereo = true);

    const data::Frame* GetLastKeyframe();

protected:
    cv::Mat keyframeImg_;
    cv::Mat keyframeStereoImg_;
    int keyframeId_;
    int prevId_;
    const data::Frame *prevFrame_;
    const data::Frame *keyframe_;

private:
    virtual int DoTrack(std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, std::vector<double> &errors, bool stereo) = 0;

    int frameNum_{0};
    const int keyframeInterval_;
};

}
}

#endif /* _TRACKER_H_ */
