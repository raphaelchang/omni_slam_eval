#include "tracking_module.h"

#include "util/tf_util.h"
#include "util/math_util.h"
#include <omp.h>

using namespace std;

namespace omni_slam
{
namespace module
{

TrackingModule::TrackingModule(std::unique_ptr<feature::Detector> &detector, std::unique_ptr<feature::Tracker> &tracker, int minFeaturesRegion)
    : detector_(std::move(detector)),
    tracker_(std::move(tracker)),
    minFeaturesRegion_(minFeaturesRegion)
{
}

TrackingModule::TrackingModule(std::unique_ptr<feature::Detector> &&detector, std::unique_ptr<feature::Tracker> &&tracker, int minFeaturesRegion)
    : TrackingModule(detector, tracker, minFeaturesRegion)
{
}

void TrackingModule::Update(std::unique_ptr<data::Frame> &frame)
{
    frames_.push_back(std::move(frame));

    int imsize = max(frames_.back()->GetImage().rows, frames_.back()->GetImage().cols);
    if (frameNum_ == 0)
    {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rs_.size() - 1; i++)
        {
            for (int j = 0; j < ts_.size() - 1; j++)
            {
                detector_->DetectInRadialRegion(*frames_.back(), landmarks_, rs_[i] * imsize, rs_[i+1] * imsize, ts_[j], ts_[j+1]);
            }
        }
        tracker_->Init(*frames_.back());

        visualization_.Init(frames_.back()->GetImage().size(), landmarks_.size());

        frameNum_++;
        return;
    }

    tracker_->Track(landmarks_, *frames_.back());

    int i = 0;
    int numGood = 0;
    map<pair<int, int>, int> regionCount;
    stats_.trackLengths.resize(landmarks_.size(), 0);
    for (data::Landmark& landmark : landmarks_)
    {
        const data::Feature *obs = landmark.GetObservationByFrameID(frames_.back()->GetID());
        if (obs != nullptr)
        {
            double x = obs->GetKeypoint().pt.x - frames_.back()->GetImage().cols / 2. + 0.5;
            double y = obs->GetKeypoint().pt.y - frames_.back()->GetImage().rows / 2. + 0.5;
            double r = sqrt(x * x + y * y) / imsize;
            double t = util::MathUtil::FastAtan2(y, x);
            vector<double>::const_iterator ri = upper_bound(rs_.begin(), rs_.end(), r);
            vector<double>::const_iterator ti = upper_bound(ts_.begin(), ts_.end(), t);
            int rinx = min((int)(ri - rs_.begin()), (int)(rs_.size() - 1)) - 1;
            int tinx = min((int)(ti - ts_.begin()), (int)(ts_.size() - 1)) - 1;
            regionCount[{rinx, tinx}]++;

            Vector2d pixelGnd;
            if (frames_.back()->GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(frames_.back()->GetInversePose(), landmark.GetGroundTruth())), pixelGnd))
            {
                Vector2d pixel;
                pixel << obs->GetKeypoint().pt.x, obs->GetKeypoint().pt.y;
                double error = (pixel - pixelGnd).norm();

                const data::Feature *obsPrev = landmark.GetObservationByFrameID((*next(frames_.rbegin()))->GetID());
                visualization_.AddTrack(cv::Point2f(pixelGnd(0), pixelGnd(1)), obsPrev->GetKeypoint().pt, obs->GetKeypoint().pt, error, i);

                double xg = pixelGnd(0) - frames_.back()->GetImage().cols / 2. + 0.5;
                double yg = pixelGnd(1) - frames_.back()->GetImage().rows / 2. + 0.5;
                double rg = sqrt(xg * xg + yg * yg) / imsize;
                stats_.radialErrors.emplace_back(vector<double>{rg, error});
                stats_.frameErrors.emplace_back(vector<double>{(double)landmark.GetNumObservations() - 1, (double)i, rg, error});
            }
            stats_.trackLengths[i]++;
            numGood++;
        }
        else
        {
            const data::Feature *obs = landmark.GetObservationByFrameID((*next(frames_.rbegin()))->GetID());
            if (obs != nullptr) // Failed in current frame
            {

                Vector2d pixelGnd;
                if (frames_.back()->GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(frames_.back()->GetInversePose(), landmark.GetGroundTruth())), pixelGnd))
                {
                    double x = pixelGnd(0) - frames_.back()->GetImage().cols / 2. + 0.5;
                    double y = pixelGnd(1) - frames_.back()->GetImage().rows / 2. + 0.5;
                    double r = sqrt(x * x + y * y) / imsize;
                    stats_.failureRadDists.push_back(r);
                }
                stats_.trackLengths.push_back(landmark.GetNumObservations() - 1);
            }
        }
        i++;
    }
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rs_.size() - 1; i++)
    {
        for (int j = 0; j < ts_.size() - 1; j++)
        {
            if (regionCount[{i, j}] < minFeaturesRegion_)
            {
                detector_->DetectInRadialRegion(*frames_.back(), landmarks_, rs_[i] * imsize, rs_[i+1] * imsize, ts_[j], ts_[j+1]);
            }
        }
    }

    stats_.frameTrackCounts.emplace_back(vector<int>{frameNum_, numGood});

    (*next(frames_.rbegin()))->CompressImages();

    frameNum_++;
}

TrackingModule::Stats& TrackingModule::GetStats()
{
    return stats_;
}

void TrackingModule::Visualize(cv::Mat &base_img)
{
    visualization_.Draw(base_img);
}

void TrackingModule::Visualization::Init(cv::Size img_size, int num_colors)
{
    visMask_ = cv::Mat::zeros(img_size, CV_8UC3);
    curMask_ = cv::Mat::zeros(img_size, CV_8UC3);
    cv::RNG rng(123);
    for (int i = 0; i < max(num_colors, 100); i++)
    {
        colors_.emplace_back(rng.uniform(10, 200), rng.uniform(10, 200), rng.uniform(10, 200));
    }
}

void TrackingModule::Visualization::AddTrack(cv::Point2f gnd, cv::Point2f prev, cv::Point2f cur, double error, int index)
{
    int maxerror = min(visMask_.rows, visMask_.cols) / 100;
    cv::Scalar color(0, (int)(255 * (1 - error / maxerror)), (int)(255 * (error / maxerror)));
    cv::line(visMask_, prev, cur, color, 1);
    cv::circle(curMask_, cur, 1, color, -1);
    cv::circle(curMask_, gnd, 3, colors_[index % colors_.size()], -1);
}

void TrackingModule::Visualization::Draw(cv::Mat &img)
{
    if (img.channels() == 1)
    {
        cv::cvtColor(img, img, CV_GRAY2BGR);
    }
    curMask_.copyTo(img, curMask_);
    cv::addWeighted(img, 1, visMask_, trackOpacity_, 0, img);
    curMask_ = cv::Mat::zeros(img.size(), CV_8UC3);
    visMask_ *= trackFade_;
}

}
}
