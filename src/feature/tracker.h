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
    virtual void Init(data::Frame &init_frame) = 0;
    virtual int Track(std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, std::vector<double> &errors, bool stereo = true) = 0;
};

}
}

#endif /* _TRACKER_H_ */
