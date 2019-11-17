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
    for (int i = 0; i < feature::Region::rs.size() - 1; i++)
    {
        for (int j = 0; j < feature::Region::ts.size() - 1; j++)
        {
            feature::Detector detector(*detector_);
            detector.DetectInRadialRegion(*frames_.back(), curLandmarks, feature::Region::rs[i] * imsize, feature::Region::rs[i+1] * imsize, feature::Region::ts[j], feature::Region::ts[j+1]);
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
    vector<vector<double>> distances;
    vector<int> indices;
    map<pair<int, int>, int> numMatches = matcher_->Match(landmarks_, curLandmarks, matches, distances, indices);
    map<pair<int, int>, int> numGoodMatches;
    map<pair<int, int>, priority_queue<double>> goodDists;
    map<pair<int, int>, priority_queue<double>> badDists;
    #pragma omp parallel for
    for (auto it = matches.begin(); it < matches.end(); it++)
    {
        data::Landmark &match = *it;
        const data::Feature &curFeat = match.GetObservations()[0];
        for (int i = 1; i < match.GetNumObservations(); i++)
        {
            data::Feature &prevFeat = match.GetObservations()[i];
            Vector2d curPix;
            curPix << curFeat.GetKeypoint().pt.x, curFeat.GetKeypoint().pt.y;
            double descDist = distances[std::distance(matches.begin(), it)][i - 1];
            double x = curPix(0) - frames_.back()->GetImage().cols / 2. + 0.5;
            double y = curPix(1) - frames_.back()->GetImage().rows / 2. + 0.5;
            double r = sqrt(x * x + y * y) / imsize;
            double x_prev = prevFeat.GetKeypoint().pt.x - frames_.back()->GetImage().cols / 2. + 0.5;
            double y_prev = prevFeat.GetKeypoint().pt.y - frames_.back()->GetImage().rows / 2. + 0.5;
            double r_prev = sqrt(x_prev * x_prev + y_prev * y_prev) / imsize;
            Vector2d prevPix;
            prevPix << prevFeat.GetKeypoint().pt.x, prevFeat.GetKeypoint().pt.y;
            Vector3d ray;
            frames_.back()->GetCameraModel().UnprojectToBearing(curPix, ray);
            Vector3d rayPrev;
            frames_.back()->GetCameraModel().UnprojectToBearing(prevPix, rayPrev);
            Vector3d z;
            z << 0, 0, 1;
            double bearingAngleCur = acos(ray.normalized().dot(z));
            double bearingAnglePrev = acos(rayPrev.normalized().dot(z));
            Vector2d prevFeatPix;
            if (frames_.back()->GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(frames_.back()->GetInversePose(), prevFeat.GetWorldPoint())), prevFeatPix))
            {
                cv::KeyPoint kpt = prevFeat.GetKeypoint();
                kpt.pt = cv::Point2f(prevFeatPix(0), prevFeatPix(1));
                double error = (curPix - prevFeatPix).norm();
                double overlap = cv::KeyPoint::overlap(curFeat.GetKeypoint(), kpt);
                if (overlap > 0)
                {
                    #pragma omp critical
                    {
                        stats_.radialOverlapsErrors.emplace_back(vector<double>{r, overlap, error});
                    }
                }
                if (overlap > overlapThresh_ && error < distThresh_)
                {
                    #pragma omp critical
                    {
                        numGoodMatches[{prevFeat.GetFrame().GetID(), curFeat.GetFrame().GetID()}]++;
                        stats_.goodRadialDistances.emplace_back(vector<double>{fabs(r - r_prev), fabs(bearingAngleCur - bearingAnglePrev), descDist});
                        goodDists[{prevFeat.GetFrame().GetID(), curFeat.GetFrame().GetID()}].push(descDist);
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
                        badDists[{prevFeat.GetFrame().GetID(), curFeat.GetFrame().GetID()}].push(descDist);
                        if (frameIdToNum_[prevFeat.GetFrame().GetID()] == 0)
                        {
                            visualization_.AddBadMatch(curFeat.GetKeypoint(), kpt);
                        }
                        stats_.badRadialDistances.emplace_back(vector<double>{fabs(r - r_prev), fabs(bearingAngleCur - bearingAnglePrev), descDist});
                    }
                }
            }
            else
            {
                #pragma omp critical
                {
                    badDists[{prevFeat.GetFrame().GetID(), curFeat.GetFrame().GetID()}].push(descDist);
                    if (frameIdToNum_[prevFeat.GetFrame().GetID()] == 0)
                    {
                        visualization_.AddBadMatch(curFeat.GetKeypoint());
                    }
                    stats_.badRadialDistances.emplace_back(vector<double>{fabs(r - r_prev), fabs(bearingAngleCur - bearingAnglePrev), descDist});
                }
            }
        }
    }
    unordered_map<int, Vector2d> projMap;
    unordered_map<int, int> featuresInId;
    #pragma omp parallel for
    for (auto it = landmarks_.begin(); it < landmarks_.end(); it++)
    {
        data::Landmark &prevLandmark = *it;
        const data::Feature &prevFeat = prevLandmark.GetObservations()[0];
        #pragma omp critical
        {
            featuresInId[prevFeat.GetFrame().GetID()]++;
        }
        Vector2d prevFeatPix;
        if (frames_.back()->GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(frames_.back()->GetInversePose(), prevLandmark.GetGroundTruth())), prevFeatPix))
        {
            #pragma omp critical
            {
                projMap[prevLandmark.GetID()] = prevFeatPix;
            }
        }
    }
    map<pair<int, int>, unordered_set<data::Landmark*>> goodCorrPrev;
    map<pair<int, int>, unordered_set<data::Landmark*>> goodCorrCur;
    #pragma omp parallel for collapse(2)
    for (auto it1 = curLandmarks.begin(); it1 < curLandmarks.end(); it1++)
    {
        for (auto it2 = landmarks_.begin(); it2 < landmarks_.end(); it2++)
        {
            data::Landmark &curLandmark = *it1;
            data::Landmark &prevLandmark = *it2;
            const data::Feature &curFeat = curLandmark.GetObservations()[0];
            const data::Feature &prevFeat = prevLandmark.GetObservations()[0];
            if (projMap.find(prevLandmark.GetID()) != projMap.end())
            {
                Vector2d prevFeatPix = projMap.at(prevLandmark.GetID());
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
    map<pair<int, int>, int> maxCorr;
    #pragma omp parallel for
    for (auto it2 = landmarks_.begin(); it2 < landmarks_.end(); it2++)
    {
        data::Landmark &prevLandmark = *it2;
        const data::Feature &prevFeat = prevLandmark.GetObservations()[0];
        Vector2d prevFeatPix;
        if (frames_.back()->GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(frames_.back()->GetInversePose(), prevLandmark.GetGroundTruth())), prevFeatPix))
        {
            #pragma omp critical
            {
                maxCorr[{prevFeat.GetFrame().GetID(), frames_.back()->GetID()}]++;
            }
        }
    }
    map<pair<int, int>, int> numCorr;
    map<pair<int, int>, int> numNeg;
    map<pair<int, int>, vector<pair<double, double>>> rocCurves;
    map<pair<int, int>, vector<pair<double, double>>> prCurves;
    for (auto &corrPrevPair : goodCorrPrev)
    {
        unordered_set<data::Landmark*> &corrPrev = corrPrevPair.second;
        unordered_set<data::Landmark*> &corrCur = goodCorrCur[corrPrevPair.first];
        numCorr[corrPrevPair.first] = min(corrPrev.size(), corrCur.size());
        numNeg[corrPrevPair.first] = featuresInId[corrPrevPair.first.first] * curLandmarks.size() - numCorr[corrPrevPair.first];
        priority_queue<double> &goodDist = goodDists[corrPrevPair.first];
        priority_queue<double> &badDist = badDists[corrPrevPair.first];
        while (!goodDist.empty() || !badDist.empty())
        {
            if (goodDist.empty())
            {
                rocCurves[corrPrevPair.first].push_back({0, (double)badDist.size() / numNeg[corrPrevPair.first]});
                prCurves[corrPrevPair.first].push_back({(double)goodDist.size() / (goodDist.size() + badDist.size()), (double)goodDist.size() / numCorr[corrPrevPair.first]});
                badDist.pop();
                continue;
            }
            if (badDist.empty())
            {
                rocCurves[corrPrevPair.first].push_back({(double)goodDist.size() / numCorr[corrPrevPair.first], 0});
                prCurves[corrPrevPair.first].push_back({(double)goodDist.size() / (goodDist.size() + badDist.size()), (double)goodDist.size() / numCorr[corrPrevPair.first]});
                goodDist.pop();
                continue;
            }
            rocCurves[corrPrevPair.first].push_back({(double)goodDist.size() / numCorr[corrPrevPair.first], (double)badDist.size() / numNeg[corrPrevPair.first]});
            prCurves[corrPrevPair.first].push_back({(double)goodDist.size() / (goodDist.size() + badDist.size()), (double)goodDist.size() / numCorr[corrPrevPair.first]});
            if (goodDist.top() < badDist.top())
            {
                badDist.pop();
            }
            else if (goodDist.top() > badDist.top())
            {
                goodDist.pop();
            }
            else
            {
                goodDist.pop();
                badDist.pop();
            }
        }
    }

    for (auto &statPair : numGoodMatches)
    {
        int frameDiff = frameIdToNum_[statPair.first.second] - frameIdToNum_[statPair.first.first];
        stats_.frameMatchStats.emplace_back(vector<double>{(double)frameDiff, (double)statPair.second, (double)statPair.second / numMatches[statPair.first], (double)statPair.second / numCorr[statPair.first], (double)numCorr[statPair.first] / maxCorr[statPair.first]});
        for (pair<double, double> &roc : rocCurves[statPair.first])
        {
            stats_.rocCurves.emplace_back(vector<double>{(double)frameIdToNum_[statPair.first.first], (double)frameIdToNum_[statPair.first.second], roc.first, roc.second});
        }
        for (pair<double, double> &pr : prCurves[statPair.first])
        {
            stats_.precRecCurves.emplace_back(vector<double>{(double)frameIdToNum_[statPair.first.first], (double)frameIdToNum_[statPair.first.second], pr.first, pr.second});
        }
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

void MatchingModule::Visualization::AddBadMatch(cv::KeyPoint query_kpt)
{
    cv::Mat kptMat;
    cv::drawKeypoints(curMask_, vector<cv::KeyPoint>{query_kpt}, kptMat, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
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
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    cv::Mat mask;
    cv::cvtColor(curMask_, mask, cv::COLOR_BGR2GRAY);
    curMask_.copyTo(img, mask);
    curMask_ = cv::Mat::zeros(img.size(), CV_8UC3);
}

}
}
