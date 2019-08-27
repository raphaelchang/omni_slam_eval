#include "detector.h"

namespace omni_slam
{
namespace feature
{

Detector::Detector(std::string type, std::map<std::string, double> args)
{
    if (type == "GFTT")
    {
        if (args.find("qualityLevel") == args.end())
        {
            detector_ = cv::GFTTDetector::create((int)args["maxCorners"]);
        }
        else if (args.find("minDistance") == args.end())
        {
            detector_ = cv::GFTTDetector::create((int)args["maxCorners"], args["qualityLevel"]);
        }
        else if (args.find("blockSize") == args.end())
        {
            detector_ = cv::GFTTDetector::create((int)args["maxCorners"], args["qualityLevel"], (int)args["minDistance"]);
        }
        else
        {
            detector_ = cv::GFTTDetector::create((int)args["maxCorners"], args["qualityLevel"], (int)args["minDistance"], (int)args["blockSize"]);
        }
    }
    else if (type == "FAST")
    {
        if (args.find("threshold") == args.end())
        {
            detector_ = cv::FastFeatureDetector::create();
        }
        else
        {
            detector_ = cv::FastFeatureDetector::create((int)args["threshold"]);
        }
    }
    else if (type == "SIFT")
    {
        if (args.find("nOctaveLayers") == args.end())
        {
            detector_ = cv::xfeatures2d::SIFT::create((int)args["nfeatures"]);
        }
        else if (args.find("contrastThreshold") == args.end())
        {
            detector_ = cv::xfeatures2d::SIFT::create((int)args["nfeatures"], (int)args["nOctaveLayers"]);
        }
        else if (args.find("edgeThreshold") == args.end())
        {
            detector_ = cv::xfeatures2d::SIFT::create((int)args["nfeatures"], (int)args["nOctaveLayers"], args["contrastThreshold"]);
        }
        else
        {
            detector_ = cv::xfeatures2d::SIFT::create((int)args["nfeatures"], (int)args["nOctaveLayers"], args["contrastThreshold"], args["edgeThreshold"]);
        }
    }
    else if (type == "SURF")
    {
        if (args.find("hessianThreshold") == args.end())
        {
            detector_ = cv::xfeatures2d::SURF::create();
        }
        else if (args.find("nOctaves") == args.end())
        {
            detector_ = cv::xfeatures2d::SURF::create(args["hessianThreshold"]);
        }
        else if (args.find("nOctaveLayers") == args.end())
        {
            detector_ = cv::xfeatures2d::SURF::create(args["hessianThreshold"], (int)args["nOctaves"]);
        }
        else
        {
            detector_ = cv::xfeatures2d::SURF::create(args["hessianThreshold"], (int)args["nOctave"], (int)args["nOctaveLayers"]);
        }
    }
    else if (type == "ORB")
    {
        if (args.find("scaleFactor") == args.end())
        {
            detector_ = cv::ORB::create((int)args["nfeatures"]);
        }
        else if (args.find("nlevels") == args.end())
        {
            detector_ = cv::ORB::create((int)args["nfeatures"], (int)args["scaleFactor"]);
        }
        else if (args.find("edgeThreshold") == args.end())
        {
            detector_ = cv::ORB::create((int)args["nfeatures"], (int)args["scaleFactor"], (int)args["nlevels"]);
        }
        else
        {
            detector_ = cv::ORB::create((int)args["nfeatures"], (int)args["scaleFactor"], (int)args["nlevels"], (int)args["edgeThreshold"]);
        }
    }
    else if (type == "BRISK")
    {
        if (args.find("thresh") == args.end())
        {
            detector_ = cv::BRISK::create();
        }
        else if (args.find("octaves") == args.end())
        {
            detector_ = cv::BRISK::create((int)args["thresh"]);
        }
        else if (args.find("patternScale") == args.end())
        {
            detector_ = cv::BRISK::create((int)args["thresh"], (int)args["octaves"]);
        }
        else
        {
            detector_ = cv::BRISK::create((int)args["thresh"], (int)args["octaves"], args["patternScale"]);
        }
    }
    else if (type == "STAR")
    {
        if (args.find("maxSize") == args.end())
        {
            detector_ = cv::xfeatures2d::StarDetector::create();
        }
        else if (args.find("responseThrehsold") == args.end())
        {
            detector_ = cv::xfeatures2d::StarDetector::create((int)args["maxSize"]);
        }
        else
        {
            detector_ = cv::xfeatures2d::StarDetector::create((int)args["maxSize"], (int)args["responseThreshold"]);
        }
    }
    else if (type == "AKAZE")
    {
        if (args.find("threshold") == args.end())
        {
            detector_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3);
        }
        else if (args.find("nOctaves") == args.end())
        {
            detector_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, args["threshold"]);
        }
        else if (args.find("nOctavesLayers") == args.end())
        {
            detector_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, args["threshold"], (int)args["nOctaves"]);
        }
        else
        {
            detector_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, args["threshold"], (int)args["nOctaves"], (int)args["nOctaveLayers"]);
        }
    }
    else if (type == "KAZE")
    {
        if (args.find("threshold") == args.end())
        {
            detector_ = cv::KAZE::create(false, false);
        }
        else if (args.find("nOctaves") == args.end())
        {
            detector_ = cv::KAZE::create(false, false, args["threshold"]);
        }
        else if (args.find("nOctavesLayers") == args.end())
        {
            detector_ = cv::KAZE::create(false, false, args["threshold"], (int)args["nOctaves"]);
        }
        else
        {
            detector_ = cv::KAZE::create(false, false, args["threshold"], (int)args["nOctaves"], (int)args["nOctaveLayers"]);
        }
    }
    else if (type == "AGAST")
    {
        if (args.find("threshold") == args.end())
        {
            detector_ = cv::AgastFeatureDetector::create();
        }
        else
        {
            detector_ = cv::AgastFeatureDetector::create((int)args["threshold"]);
        }
    }
}

int Detector::Detect(data::Frame &frame, std::vector<data::Landmark> &landmarks)
{
    cv::Mat noarr;
    return DetectInRegion(frame, landmarks, noarr);
}

int Detector::DetectInRectangularRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Point2f start, cv::Point2f end)
{
    cv::Mat mask = cv::Mat::zeros(frame.GetImage().size(), CV_8U);
    mask(cv::Rect(start, end)) = 255;
    return DetectInRegion(frame, landmarks, mask);
}

int Detector::DetectInRadialRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, double start_r, double end_r, double start_t, double end_t)
{
    cv::Mat mask = cv::Mat::zeros(frame.GetImage().size(), CV_8U);
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            double x = j - mask.cols / 2. + 0.5;
            double y = i - mask.rows / 2. + 0.5;
            double r = sqrt(x * x + y * y);
            double t = atan2(y, x);
            if (r >= start_r && r < end_r && t >= start_t && t < end_t)
            {
                mask.at<unsigned char>(i, j) = 255;
            }
        }
    }
    //cv::imshow("mask", mask);
    //cv::waitKey(0);
    return DetectInRegion(frame, landmarks, mask);
}

int Detector::DetectInRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Mat &mask)
{
    bool compressed = frame.IsCompressed();
    std::vector<cv::KeyPoint> kpts;
    detector_->detect(frame.GetImage(), kpts, mask);
    for (cv::KeyPoint kpt : kpts)
    {
        data::Feature feat(frame, kpt);
        data::Landmark landmark;
        landmark.AddObservation(feat);
        landmarks.push_back(landmark);
    }
    if (compressed)
    {
        frame.CompressImages();
    }
    return kpts.size();
}

bool Detector::IsDetectorTypeValid(std::string name)
{
    std::vector<std::string> valid = {"GFTT", "FAST", "SIFT", "SURF", "ORB", "BRISK", "STAR", "AKAZE", "KAZE", "AGAST"};
    return std::find(valid.begin(), valid.end(), name) != valid.end();
}

}
}
