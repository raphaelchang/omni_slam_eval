#include "matcher.h"

namespace omni_slam
{
namespace feature
{

Matcher::Matcher(std::string descriptor_type, double max_dist)
    : maxDistance_(max_dist)
{
    if (descriptor_type == "SIFT" || descriptor_type == "SURF" || descriptor_type == "KAZE" || descriptor_type == "DAISY" || descriptor_type == "VGG")
    {
        matcher_ = cv::BFMatcher::create(cv::NORM_L2, true);
    }
    else if (descriptor_type == "ORB" || descriptor_type == "BRISK" || descriptor_type == "AKAZE" || descriptor_type == "FREAK" || descriptor_type == "LATCH" || descriptor_type == "LUCID" || descriptor_type == "BOOST")
    {
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    }
}

std::map<std::pair<int, int>, int> Matcher::Match(const std::vector<data::Landmark> &train, const std::vector<data::Landmark> &query, std::vector<data::Landmark> &matches, std::vector<std::vector<double>> &distances, std::vector<int> &query_match_indices, bool stereo) const
{
    std::map<int, cv::Mat> trainDescriptors;
    std::map<int, std::vector<const data::Feature*>> trainDescInxToFeature;
    for (const data::Landmark &landmark : train)
    {
        for (const data::Feature &feat : stereo ? landmark.GetStereoObservations() : landmark.GetObservations())
        {
            const int id = feat.GetFrame().GetID();
            trainDescInxToFeature[id].push_back(&feat);
            trainDescriptors[id].push_back(feat.GetDescriptor());
        }
    }
    std::map<int, cv::Mat> queryDescriptors;
    std::map<int, std::vector<const data::Feature*>> queryDescInxToFeature;
    std::map<const data::Feature*, int> featureToQueryInx;
    int i = 0;
    for (const data::Landmark &landmark : query)
    {
        for (const data::Feature &feat : stereo ? landmark.GetStereoObservations() : landmark.GetObservations())
        {
            const int id = feat.GetFrame().GetID();
            queryDescInxToFeature[id].push_back(&feat);
            queryDescriptors[id].push_back(feat.GetDescriptor());
            featureToQueryInx[&feat] = i;
        }
        i++;
    }

    std::map<std::pair<int, int>, int> numMatches;
    matches.clear();
    distances.clear();
    query_match_indices.clear();
    std::map<const data::Feature*, int> featureToMatchesInx;
    for (auto &queryPair : queryDescriptors)
    {
        for (auto &trainPair : trainDescriptors)
        {
            if (trainPair.first != queryPair.first)
            {
                std::vector<cv::DMatch> dmatches;
                matcher_->match(queryPair.second, trainPair.second, dmatches);
                int numGood = 0;
                for (cv::DMatch &match : dmatches)
                {
                    double dist = match.distance;
                    if (maxDistance_ > 0 && dist > maxDistance_)
                    {
                        continue;
                    }
                    const data::Feature *queryFeat = queryDescInxToFeature[queryPair.first][match.queryIdx];
                    if (featureToMatchesInx.find(queryFeat) == featureToMatchesInx.end())
                    {
                        featureToMatchesInx[queryFeat] = matches.size();
                        data::Landmark landmark;
                        landmark.AddObservation(*queryFeat, false);
                        matches.push_back(landmark);
                        distances.push_back(std::vector<double>());
                        query_match_indices.push_back(featureToQueryInx[queryFeat]);
                    }
                    matches[featureToMatchesInx[queryFeat]].AddObservation(*trainDescInxToFeature[trainPair.first][match.trainIdx]);
                    distances[featureToMatchesInx[queryFeat]].push_back(fabs(match.distance));
                    numGood++;
                }
                numMatches[{trainPair.first, queryPair.first}] = numGood;
            }
        }
    }
    return numMatches;
}

}
}
