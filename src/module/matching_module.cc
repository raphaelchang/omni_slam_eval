#include "matching_module.h"

#include "util/tf_util.h"

using namespace std;

namespace omni_slam
{
namespace module
{

MatchingModule::MatchingModule(std::unique_ptr<feature::Detector> &detector, std::unique_ptr<feature::Matcher> &matcher, double overlap_thresh, double dist_thresh)
    : detector_(std::move(detector)),
    matcher_(std::move(matcher)),
    overlapThresh_(overlap_thresh),
    distThresh_(dist_thresh)
{
}

MatchingModule::MatchingModule(std::unique_ptr<feature::Detector> &&detector, std::unique_ptr<feature::Matcher> &&matcher, double overlap_thresh, double dist_thresh)
    : MatchingModule(detector, matcher, overlap_thresh, dist_thresh)
{
}

void MatchingModule::Update(std::unique_ptr<data::Frame> &frame)
{
    frames_.push_back(std::move(frame));
    frameIdToNum_[frames_.back()->GetID()] = frameNum_;

    vector<data::Landmark> curLandmarks;
    int imsize = max(frames_.back()->GetImage().rows, frames_.back()->GetImage().cols);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rs_.size() - 1; i++)
    {
        for (int j = 0; j < ts_.size() - 1; j++)
        {   
            feature::Detector detector(*detector_);
            detector.DetectInRadialRegion(*frames_.back(), curLandmarks, rs_[i] * imsize, rs_[i+1] * imsize, ts_[j], ts_[j+1]);
        }
    }

    if (frameNum_ == 0)
    {
        visualization_.Init(frames_.back()->GetImage().size());
    }

    vector<cv::KeyPoint> drawKpts;
    for (data::Landmark &landmark : curLandmarks)
    {
        drawKpts.push_back(landmark.GetObservations()[0].GetKeypoint());
    }
    visualization_.AddDetections(drawKpts);

    if (frameNum_ == 0)
    {
        landmarks_ = std::move(curLandmarks);
        frames_.back()->CompressImages();
        frameNum_++;
        return;
    }

    vector<data::Landmark> matches;
    map<pair<int, int>, int> numMatches = matcher_->Match(landmarks_, curLandmarks, matches);
    map<pair<int, int>, int> numGoodMatches;
    #pragma omp parallel for
    for (auto it = matches.begin(); it < matches.end(); it++)
    {
        data::Landmark &match = *it;
        const data::Feature &curFeat = match.GetObservations()[0];
        for (int i = 1; i < match.GetNumObservations(); i++)
        {
            data::Feature &prevFeat = match.GetObservations()[i];
            Vector2d prevFeatPix;
            if (frames_.back()->GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(frames_.back()->GetInversePose(), prevFeat.GetWorldPoint())), prevFeatPix))
            {
                cv::KeyPoint kpt = prevFeat.GetKeypoint();
                kpt.pt = cv::Point2f(prevFeatPix(0), prevFeatPix(1));
                Vector2d curPix;
                curPix << curFeat.GetKeypoint().pt.x, curFeat.GetKeypoint().pt.y;
                double error = (curPix - prevFeatPix).norm();
                double overlap = cv::KeyPoint::overlap(curFeat.GetKeypoint(), kpt);
                if (overlap > 0)
                {
                    double x = curPix(0) - frames_.back()->GetImage().cols / 2. + 0.5;
                    double y = curPix(1) - frames_.back()->GetImage().rows / 2. + 0.5;
                    double r = sqrt(x * x + y * y) / imsize;
                    #pragma omp critical
                    {
                        stats_.radialOverlapsErrors.emplace_back(vector<double>{r, overlap, error});
                    }
                }
                if (overlap > overlapThresh_ && error < distThresh_)
                {
                    double x = curPix(0) - frames_.back()->GetImage().cols / 2. + 0.5;
                    double y = curPix(1) - frames_.back()->GetImage().rows / 2. + 0.5;
                    double r = sqrt(x * x + y * y) / imsize;
                    double x_prev = prevFeat.GetKeypoint().pt.x - frames_.back()->GetImage().cols / 2. + 0.5;
                    double y_prev = prevFeat.GetKeypoint().pt.y - frames_.back()->GetImage().rows / 2. + 0.5;
                    double r_prev = sqrt(x_prev * x_prev + y_prev * y_prev) / imsize;
                    #pragma omp critical
                    {
                        numGoodMatches[{prevFeat.GetFrame().GetID(), curFeat.GetFrame().GetID()}]++;
                        stats_.deltaRadius.push_back(fabs(r - r_prev));
                        if (frameIdToNum_[prevFeat.GetFrame().GetID()] == 0)
                        {
                            visualization_.AddGoodMatch(curFeat.GetKeypoint(), kpt, overlap);
                        }
                    }
                }
                else
                {
                    #pragma omp critical
                    {
                        if (frameIdToNum_[prevFeat.GetFrame().GetID()] == 0)
                        {
                            visualization_.AddBadMatch(curFeat.GetKeypoint(), kpt);
                        }
                    }
                }
            }
        }
    }
    map<pair<int, int>, set<data::Landmark*>> goodCorrPrev;
    map<pair<int, int>, set<data::Landmark*>> goodCorrCur;
    #pragma omp parallel for collapse(2)
    for (auto it1 = curLandmarks.begin(); it1 < curLandmarks.end(); it1++)
    {
        for (auto it2 = landmarks_.begin(); it2 < landmarks_.end(); it2++)
        {
            data::Landmark &curLandmark = *it1;
            data::Landmark &prevLandmark = *it2;
            const data::Feature &curFeat = curLandmark.GetObservations()[0];
            const data::Feature &prevFeat = prevLandmark.GetObservations()[0];
            Vector2d prevFeatPix;
            if (frames_.back()->GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(frames_.back()->GetInversePose(), prevLandmark.GetGroundTruth())), prevFeatPix))
            {
                cv::KeyPoint kpt = prevFeat.GetKeypoint();
                kpt.pt = cv::Point2f(prevFeatPix(0), prevFeatPix(1));
                Vector2d curPix;
                curPix << curFeat.GetKeypoint().pt.x, curFeat.GetKeypoint().pt.y;
                double error = (curPix - prevFeatPix).norm();
                if (cv::KeyPoint::overlap(curFeat.GetKeypoint(), kpt) > overlapThresh_ && error < distThresh_)
                {
                    #pragma omp critical
                    {
                        goodCorrPrev[{prevFeat.GetFrame().GetID(), curFeat.GetFrame().GetID()}].insert(&prevLandmark);
                        goodCorrCur[{prevFeat.GetFrame().GetID(), curFeat.GetFrame().GetID()}].insert(&curLandmark);
                        if (frameIdToNum_[prevFeat.GetFrame().GetID()] == 0)
                        {
                            visualization_.AddCorrespondence(curFeat.GetKeypoint(), kpt);
                        }
                    }
                }
            }
        }
    }
    map<pair<int, int>, int> numCorr;
    for (auto &corrPrevPair : goodCorrPrev)
    {
        set<data::Landmark*> &corrPrev = corrPrevPair.second;
        set<data::Landmark*> &corrCur = goodCorrCur[corrPrevPair.first];
        numCorr[corrPrevPair.first] = min(corrPrev.size(), corrCur.size());
    }

    for (auto &statPair : numGoodMatches)
    {
        int frameDiff = frameIdToNum_[statPair.first.second] - frameIdToNum_[statPair.first.first];
        stats_.frameMatchStats.emplace_back(vector<double>{(double)frameDiff, (double)statPair.second, (double)statPair.second / numMatches[statPair.first], (double)statPair.second / numCorr[statPair.first]});
    }

    //landmarks_.reserve(landmarks_.size() + curLandmarks.size());
    //std::move(std::begin(curLandmarks), std::end(curLandmarks), std::back_inserter(landmarks_));
    frames_.back()->CompressImages();
    frameNum_++;
}

MatchingModule::Stats& MatchingModule::GetStats()
{
    return stats_;
}

void MatchingModule::Visualize(cv::Mat &base_img)
{
    visualization_.Draw(base_img);
}

void MatchingModule::Visualization::Init(cv::Size img_size)
{
    curMask_ = cv::Mat::zeros(img_size, CV_8UC3);
}

void MatchingModule::Visualization::AddDetections(vector<cv::KeyPoint> kpt)
{
    cv::Mat kptMat;
    cv::drawKeypoints(curMask_, kpt, kptMat, cv::Scalar(100, 0, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    curMask_ = kptMat;
}

void MatchingModule::Visualization::AddGoodMatch(cv::KeyPoint query_kpt, cv::KeyPoint train_kpt, double overlap)
{
    cv::Mat kptMat;
    cv::Scalar color(0, (int)(255 * overlap), (int)(255 * (1 - overlap)));
    cv::drawKeypoints(curMask_, vector<cv::KeyPoint>{query_kpt}, kptMat, color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::circle(kptMat, train_kpt.pt, 4, color, -1);
    cv::line(kptMat, query_kpt.pt, train_kpt.pt, color, 2);
    curMask_ = kptMat;
}

void MatchingModule::Visualization::AddBadMatch(cv::KeyPoint query_kpt, cv::KeyPoint train_kpt)
{
    cv::Mat kptMat;
    cv::drawKeypoints(curMask_, vector<cv::KeyPoint>{query_kpt}, kptMat, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    //cv::circle(kptMat, train_kpt.pt, 2, cv::Scalar(0, 0, 255), -1);
    cv::line(kptMat, query_kpt.pt, train_kpt.pt, cv::Scalar(0, 0, 255), 1);
    curMask_ = kptMat;
}

void MatchingModule::Visualization::AddCorrespondence(cv::KeyPoint query_kpt, cv::KeyPoint train_kpt)
{
    cv::circle(curMask_, train_kpt.pt, 3, cv::Scalar(255, 0, 0), -1);
    cv::line(curMask_, query_kpt.pt, train_kpt.pt, cv::Scalar(255, 0, 0), 2);
}

void MatchingModule::Visualization::Draw(cv::Mat &img)
{
    if (img.channels() == 1)
    {
        cv::cvtColor(img, img, CV_GRAY2BGR);
    }
    cv::Mat mask;
    cv::cvtColor(curMask_, mask, CV_BGR2GRAY);
    curMask_.copyTo(img, mask);
    curMask_ = cv::Mat::zeros(img.size(), CV_8UC3);
}

}
}
