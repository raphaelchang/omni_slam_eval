#ifndef _MATCHING_MODULE_H_
#define _MATCHING_MODULE_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

#include "feature/matcher.h"
#include "feature/detector.h"
#include "data/frame.h"
#include "data/landmark.h"

namespace omni_slam
{
namespace module
{

class MatchingModule
{
public:
    struct Stats
    {
        std::vector<std::vector<double>> frameMatchStats;
        std::vector<std::vector<double>> rocCurves;
        std::vector<std::vector<double>> precRecCurves;
        std::vector<std::vector<double>> radialOverlapsErrors;
        std::vector<std::vector<double>> goodRadialDistances;
        std::vector<std::vector<double>> badRadialDistances;
    };

    MatchingModule(std::unique_ptr<feature::Detector> &detector, std::unique_ptr<feature::Matcher> &matcher, double overlap_thresh = 0.5, double dist_thresh = 10.);
    MatchingModule(std::unique_ptr<feature::Detector> &&detector, std::unique_ptr<feature::Matcher> &&matcher, double overlap_thresh = 0.5, double dist_thresh = 10.);

    void Update(std::unique_ptr<data::Frame> &frame);

    Stats& GetStats();
    void Visualize(cv::Mat &base_img);

private:
    class Visualization
    {
    public:
        void Init(cv::Size img_size);
        void AddDetections(std::vector<cv::KeyPoint> kpt);
        void AddGoodMatch(cv::KeyPoint query_kpt, cv::KeyPoint train_kpt, double overlap);
        void AddBadMatch(cv::KeyPoint query_kpt, cv::KeyPoint train_kpt);
        void AddCorrespondence(cv::KeyPoint query_kpt, cv::KeyPoint train_kpt);
        void Draw(cv::Mat &img);

    private:
        cv::Mat curMask_;
    };

    std::shared_ptr<feature::Detector> detector_;
    std::shared_ptr<feature::Matcher> matcher_;

    std::vector<std::unique_ptr<data::Frame>> frames_;
    std::vector<data::Landmark> landmarks_;

    const std::vector<double> rs_{0, 0.1, 0.2, 0.3, 0.4, 0.5};
    const std::vector<double> ts_{-M_PI, -3 * M_PI / 4, -M_PI / 2, -M_PI / 4, 0, M_PI / 4, M_PI / 2, 3 * M_PI / 4, M_PI};

    double overlapThresh_;
    double distThresh_;

    int frameNum_{0};
    std::map<int, int> frameIdToNum_;

    Visualization visualization_;
    Stats stats_;
};

}
}

#endif /* _MATCHING_MODULE_H_ */
