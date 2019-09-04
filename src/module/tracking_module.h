#ifndef _TRACKING_MODULE_H_
#define _TRACKING_MODULE_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

#include "feature/tracker.h"
#include "feature/detector.h"
#include "data/frame.h"
#include "data/landmark.h"

namespace omni_slam
{
namespace module
{

class TrackingModule
{
public:
    struct Stats
    {
        std::vector<std::vector<double>> radialErrors;
        std::vector<std::vector<double>> frameErrors;
        std::vector<std::vector<int>> frameTrackCounts;
        std::vector<int> trackLengths;
        std::vector<double> failureRadDists;
    };

    TrackingModule(std::unique_ptr<feature::Detector> &detector, std::unique_ptr<feature::Tracker> &tracker, int minFeaturesRegion = 5);
    TrackingModule(std::unique_ptr<feature::Detector> &&detector, std::unique_ptr<feature::Tracker> &&tracker, int minFeaturesRegion = 5);

    void Update(std::unique_ptr<data::Frame> &frame);

    std::vector<data::Landmark>& GetLandmarks();

    Stats& GetStats();
    void Visualize(cv::Mat &base_img);

private:
    class Visualization
    {
    public:
        void Init(cv::Size img_size, int num_colors);
        void AddTrack(cv::Point2f gnd, cv::Point2f prev, cv::Point2f cur, double error, int index);
        void Draw(cv::Mat &img);

    private:
        cv::Mat visMask_;
        cv::Mat curMask_;
        std::vector<cv::Scalar> colors_;
        const double trackOpacity_{0.7};
        const double trackFade_{0.99};
    };

    std::shared_ptr<feature::Detector> detector_;
    std::shared_ptr<feature::Tracker> tracker_;

    std::vector<std::unique_ptr<data::Frame>> frames_;
    std::vector<data::Landmark> landmarks_;

    const std::vector<double> rs_{0, 0.1, 0.2, 0.3, 0.4, 0.49};
    const std::vector<double> ts_{-M_PI, -3 * M_PI / 4, -M_PI / 2, -M_PI / 4, 0, M_PI / 4, M_PI / 2, 3 * M_PI / 4, M_PI};
    int minFeaturesRegion_;

    Stats stats_;
    Visualization visualization_;

    int frameNum_{0};
};

}
}

#endif /* _TRACKING_MODULE_H_ */
