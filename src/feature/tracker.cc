#include "tracker.h"
#include <cmath>
#include "util/tf_util.h"

namespace omni_slam
{
namespace feature
{

Tracker::Tracker(const int window_size, const int num_scales, const float delta_pix_err_thresh, const float err_thresh, const int term_count, const double term_eps)
    : windowSize_(window_size / pow(2, num_scales), window_size / pow(2, num_scales)),
    numScales_(num_scales),
    errThresh_(err_thresh),
    deltaPixErrThresh_(delta_pix_err_thresh),
    termCrit_(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, term_count, term_eps),
    prevFrame_(nullptr)
{
}

void Tracker::Init(data::Frame &init_frame)
{
    prevFrame_ = &init_frame;
}

int Tracker::Track(std::vector<data::Landmark> &landmarks, data::Frame &cur_frame)
{
    if (prevFrame_ == nullptr)
    {
        return 0;
    }
    bool prevCompressed = prevFrame_->IsCompressed();
    bool curCompressed = cur_frame.IsCompressed();
    int prev_id = prevFrame_->GetID();
    std::vector<cv::Point2f> points_to_track;
    std::vector<cv::KeyPoint> orig_kpt;
    std::vector<int> orig_inx;
    for (int i = 0; i < landmarks.size(); i++)
    {
        data::Landmark &landmark = landmarks[i];
        const data::Feature *feat = landmark.GetObservationByFrameID(prev_id);
        if (feat != nullptr)
        {
            points_to_track.push_back(feat->GetKeypoint().pt);
            orig_kpt.push_back(feat->GetKeypoint());
            orig_inx.push_back(i);
        }
    }
    if (points_to_track.size() == 0)
    {
        prevFrame_ = &cur_frame;
        return 0;
    }
    std::vector<cv::Point2f> results;
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevFrame_->GetImage(), cur_frame.GetImage(), points_to_track, results, status, err, windowSize_, numScales_, termCrit_, 0);
    int numGood = 0;
    for (int i = 0; i < results.size(); i++)
    {
        data::Landmark &landmark = landmarks[orig_inx[i]];
        if (landmark.HasGroundTruth() && cur_frame.HasPose())
        {
            Vector2d pixel_gnd;
            Vector2d pixel_gnd_prev;
            if (!cur_frame.GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(cur_frame.GetInversePose(), landmark.GetGroundTruth())), pixel_gnd) || !prevFrame_->GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(prevFrame_->GetInversePose(), landmark.GetGroundTruth())), pixel_gnd_prev))
            {
                continue;
            }
            else
            {
                Vector2d pixel_cur;
                pixel_cur << results[i].x, results[i].y;
                Vector2d pixel_prev;
                pixel_prev << points_to_track[i].x, points_to_track[i].y;
                double cur_error = (pixel_cur - pixel_gnd).norm();
                double prev_error = (pixel_prev - pixel_gnd_prev).norm();
                if (cur_error - prev_error > deltaPixErrThresh_)
                {
                    continue;
                }
            }
        }
        if (status[i] == 1 && err[i] <= errThresh_)
        {
            cv::KeyPoint kpt(results[i], orig_kpt[i].size);
            data::Feature feat(cur_frame, kpt);
            landmark.AddObservation(feat);
            numGood++;
        }
    }
    if (prevCompressed)
    {
        prevFrame_->CompressImages();
    }
    if (curCompressed)
    {
        cur_frame.CompressImages();
    }

    prevFrame_ = &cur_frame;

    return numGood;
}

}
}
