#include "detector.h"
#include "util/math_util.h"

namespace omni_slam
{
namespace feature
{

Detector::Detector(std::string detector_type, std::string descriptor_type, std::map<std::string, double> det_args, std::map<std::string, double> desc_args)
    : detectorType_(detector_type),
    descriptorType_(descriptor_type),
    detectorArgs_(det_args),
    descriptorArgs_(desc_args)
{
    if (detector_type == "GFTT")
    {
        if (det_args.find("qualityLevel") == det_args.end())
        {
            detector_ = cv::GFTTDetector::create((int)det_args["maxCorners"]);
        }
        else if (det_args.find("minDistance") == det_args.end())
        {
            detector_ = cv::GFTTDetector::create((int)det_args["maxCorners"], det_args["qualityLevel"]);
        }
        else if (det_args.find("blockSize") == det_args.end())
        {
            detector_ = cv::GFTTDetector::create((int)det_args["maxCorners"], det_args["qualityLevel"], (int)det_args["minDistance"]);
        }
        else
        {
            detector_ = cv::GFTTDetector::create((int)det_args["maxCorners"], det_args["qualityLevel"], (int)det_args["minDistance"], (int)det_args["blockSize"]);
        }
    }
    else if (detector_type == "FAST")
    {
        if (det_args.find("threshold") == det_args.end())
        {
            detector_ = cv::FastFeatureDetector::create();
        }
        else
        {
            detector_ = cv::FastFeatureDetector::create((int)det_args["threshold"]);
        }
    }
    else if (detector_type == "SIFT")
    {
        if (det_args.find("nOctaveLayers") == det_args.end())
        {
            detector_ = cv::xfeatures2d::SIFT::create((int)det_args["nfeatures"]);
        }
        else if (det_args.find("contrastThreshold") == det_args.end())
        {
            detector_ = cv::xfeatures2d::SIFT::create((int)det_args["nfeatures"], (int)det_args["nOctaveLayers"]);
        }
        else if (det_args.find("edgeThreshold") == det_args.end())
        {
            detector_ = cv::xfeatures2d::SIFT::create((int)det_args["nfeatures"], (int)det_args["nOctaveLayers"], det_args["contrastThreshold"]);
        }
        else if (det_args.find("sigma") == det_args.end())
        {
            detector_ = cv::xfeatures2d::SIFT::create((int)det_args["nfeatures"], (int)det_args["nOctaveLayers"], det_args["contrastThreshold"], det_args["edgeThreshold"]);
        }
        else
        {
            detector_ = cv::xfeatures2d::SIFT::create((int)det_args["nfeatures"], (int)det_args["nOctaveLayers"], det_args["contrastThreshold"], det_args["edgeThreshold"], det_args["sigma"]);
        }
    }
    else if (detector_type == "SURF")
    {
        if (det_args.find("hessianThreshold") == det_args.end())
        {
            detector_ = cv::xfeatures2d::SURF::create();
        }
        else if (det_args.find("nOctaves") == det_args.end())
        {
            detector_ = cv::xfeatures2d::SURF::create(det_args["hessianThreshold"]);
        }
        else if (det_args.find("nOctaveLayers") == det_args.end())
        {
            detector_ = cv::xfeatures2d::SURF::create(det_args["hessianThreshold"], (int)det_args["nOctaves"]);
        }
        else
        {
            detector_ = cv::xfeatures2d::SURF::create(det_args["hessianThreshold"], (int)det_args["nOctave"], (int)det_args["nOctaveLayers"]);
        }
    }
    else if (detector_type == "ORB")
    {
        if (det_args.find("scaleFactor") == det_args.end())
        {
            detector_ = cv::ORB::create((int)det_args["nfeatures"]);
        }
        else if (det_args.find("nlevels") == det_args.end())
        {
            detector_ = cv::ORB::create((int)det_args["nfeatures"], (int)det_args["scaleFactor"]);
        }
        else if (det_args.find("edgeThreshold") == det_args.end())
        {
            detector_ = cv::ORB::create((int)det_args["nfeatures"], (int)det_args["scaleFactor"], (int)det_args["nlevels"]);
        }
        else
        {
            detector_ = cv::ORB::create((int)det_args["nfeatures"], (int)det_args["scaleFactor"], (int)det_args["nlevels"], (int)det_args["edgeThreshold"]);
        }
    }
    else if (detector_type == "BRISK")
    {
        if (det_args.find("thresh") == det_args.end())
        {
            detector_ = cv::BRISK::create();
        }
        else if (det_args.find("octaves") == det_args.end())
        {
            detector_ = cv::BRISK::create((int)det_args["thresh"]);
        }
        else if (det_args.find("patternScale") == det_args.end())
        {
            detector_ = cv::BRISK::create((int)det_args["thresh"], (int)det_args["octaves"]);
        }
        else
        {
            detector_ = cv::BRISK::create((int)det_args["thresh"], (int)det_args["octaves"], det_args["patternScale"]);
        }
    }
    else if (detector_type == "STAR")
    {
        if (det_args.find("maxSize") == det_args.end())
        {
            detector_ = cv::xfeatures2d::StarDetector::create();
        }
        else if (det_args.find("responseThrehsold") == det_args.end())
        {
            detector_ = cv::xfeatures2d::StarDetector::create((int)det_args["maxSize"]);
        }
        else
        {
            detector_ = cv::xfeatures2d::StarDetector::create((int)det_args["maxSize"], (int)det_args["responseThreshold"]);
        }
    }
    else if (detector_type == "AKAZE")
    {
        if (det_args.find("threshold") == det_args.end())
        {
            detector_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3);
        }
        else if (det_args.find("nOctaves") == det_args.end())
        {
            detector_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, det_args["threshold"]);
        }
        else if (det_args.find("nOctavesLayers") == det_args.end())
        {
            detector_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, det_args["threshold"], (int)det_args["nOctaves"]);
        }
        else if (det_args.find("diffusivity") == det_args.end())
        {
            detector_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, det_args["threshold"], (int)det_args["nOctaves"], (int)det_args["nOctaveLayers"]);
        }
        else
        {
            detector_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, det_args["threshold"], (int)det_args["nOctaves"], (int)det_args["nOctaveLayers"], (cv::KAZE::DiffusivityType)det_args["diffusivity"]);
        }
    }
    else if (detector_type == "KAZE")
    {
        if (det_args.find("threshold") == det_args.end())
        {
            detector_ = cv::KAZE::create(false, false);
        }
        else if (det_args.find("nOctaves") == det_args.end())
        {
            detector_ = cv::KAZE::create(false, false, det_args["threshold"]);
        }
        else if (det_args.find("nOctavesLayers") == det_args.end())
        {
            detector_ = cv::KAZE::create(false, false, det_args["threshold"], (int)det_args["nOctaves"]);
        }
        else if (det_args.find("diffusivity") == det_args.end())
        {
            detector_ = cv::KAZE::create(false, false, det_args["threshold"], (int)det_args["nOctaves"], (int)det_args["nOctaveLayers"]);
        }
        else
        {
            detector_ = cv::KAZE::create(false, false, det_args["threshold"], (int)det_args["nOctaves"], (int)det_args["nOctaveLayers"], (cv::KAZE::DiffusivityType)det_args["diffusivity"]);
        }
    }
    else if (detector_type == "AGAST")
    {
        if (det_args.find("threshold") == det_args.end())
        {
            detector_ = cv::AgastFeatureDetector::create();
        }
        else
        {
            detector_ = cv::AgastFeatureDetector::create((int)det_args["threshold"]);
        }
    }

    if (descriptor_type == "SIFT")
    {
        if (desc_args.find("nOctaveLayers") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::SIFT::create();
        }
        else if (desc_args.find("sigma") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::SIFT::create(0, (int)desc_args["nOctaveLayers"]);
        }
        else
        {
            descriptor_ = cv::xfeatures2d::SIFT::create(0, (int)desc_args["nOctaveLayers"], 0.04, 10, desc_args["sigma"]);
        }
    }
    else if (descriptor_type == "SURF")
    {
        if (desc_args.find("nOctaves") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::SURF::create();
        }
        else if (desc_args.find("nOctaveLayers") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::SURF::create(100, (int)desc_args["nOctaves"]);
        }
        else
        {
            descriptor_ = cv::xfeatures2d::SURF::create(100, (int)desc_args["nOctave"], (int)desc_args["nOctaveLayers"]);
        }
    }
    else if (descriptor_type == "ORB")
    {
        if (desc_args.find("scaleFactor") == desc_args.end())
        {
            descriptor_ = cv::ORB::create();
        }
        else if (desc_args.find("nlevels") == desc_args.end())
        {
            descriptor_ = cv::ORB::create(500, (int)desc_args["scaleFactor"]);
        }
        else if (desc_args.find("WTA_K") == desc_args.end())
        {
            descriptor_ = cv::ORB::create(500, (int)desc_args["scaleFactor"], (int)desc_args["nlevels"]);
        }
        else if (desc_args.find("patchSize") == desc_args.end())
        {
            descriptor_ = cv::ORB::create(500, (int)desc_args["scaleFactor"], (int)desc_args["nlevels"], 31, 0, (int)desc_args["WTA_K"]);
        }
        else
        {
            descriptor_ = cv::ORB::create(500, (int)desc_args["scaleFactor"], (int)desc_args["nlevels"], 31, 0, (int)desc_args["WTA_K"], cv::ORB::HARRIS_SCORE, (int)desc_args["patchSize"]);
        }
    }
    else if (descriptor_type == "BRISK")
    {
        if (desc_args.find("octaves") == desc_args.end())
        {
            descriptor_ = cv::BRISK::create();
        }
        else if (desc_args.find("patternScale") == desc_args.end())
        {
            descriptor_ = cv::BRISK::create(30, (int)desc_args["octaves"]);
        }
        else
        {
            descriptor_ = cv::BRISK::create(30, (int)desc_args["octaves"], desc_args["patternScale"]);
        }
    }
    else if (descriptor_type == "AKAZE")
    {
        if (desc_args.find("nOctaves") == desc_args.end())
        {
            descriptor_ = cv::AKAZE::create();
        }
        else if (desc_args.find("nOctavesLayers") == desc_args.end())
        {
            descriptor_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, (int)desc_args["nOctaves"]);
        }
        else if (desc_args.find("diffusivity") == desc_args.end())
        {
            descriptor_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, (int)desc_args["nOctaves"], (int)desc_args["nOctaveLayers"]);
        }
        else
        {
            descriptor_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, (int)desc_args["nOctaves"], (int)desc_args["nOctaveLayers"], (cv::KAZE::DiffusivityType)desc_args["diffusivity"]);
        }
    }
    else if (descriptor_type == "KAZE")
    {
        if (desc_args.find("nOctaves") == desc_args.end())
        {
            descriptor_ = cv::KAZE::create();
        }
        else if (desc_args.find("nOctavesLayers") == desc_args.end())
        {
            descriptor_ = cv::KAZE::create(false, false, 0.001f, (int)desc_args["nOctaves"]);
        }
        else if (desc_args.find("diffusivity") == desc_args.end())
        {
            descriptor_ = cv::KAZE::create(false, false, desc_args["threshold"], (int)desc_args["nOctaves"], (int)desc_args["nOctaveLayers"]);
        }
        else
        {
            descriptor_ = cv::KAZE::create(false, false, desc_args["threshold"], (int)desc_args["nOctaves"], (int)desc_args["nOctaveLayers"], (cv::KAZE::DiffusivityType)desc_args["diffusivity"]);
        }
    }
    else if (descriptor_type == "DAISY")
    {
        if (desc_args.find("radius") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::DAISY::create(15, 3, 8, 8, cv::xfeatures2d::DAISY::NRM_NONE, cv::noArray(), true, true);
        }
        else if (desc_args.find("q_radius") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::DAISY::create((int)desc_args["radius"], 3, 8, 8, cv::xfeatures2d::DAISY::NRM_NONE, cv::noArray(), true, true);
        }
        else if (desc_args.find("q_theta") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::DAISY::create((int)desc_args["radius"], (int)desc_args["q_radius"], 8, 8, cv::xfeatures2d::DAISY::NRM_NONE, cv::noArray(), true, true);
        }
        else if (desc_args.find("q_hist") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::DAISY::create((int)desc_args["radius"], (int)desc_args["q_radius"], (int)desc_args["q_theta"], 8, cv::xfeatures2d::DAISY::NRM_NONE, cv::noArray(), true, true);
        }
        else if (desc_args.find("norm") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::DAISY::create((int)desc_args["radius"], (int)desc_args["q_radius"], (int)desc_args["q_theta"], (int)desc_args["q_hist"], cv::xfeatures2d::DAISY::NRM_NONE, cv::noArray(), true, true);
        }
        else
        {
            descriptor_ = cv::xfeatures2d::DAISY::create((int)desc_args["radius"], (int)desc_args["q_radius"], (int)desc_args["q_theta"], (int)desc_args["q_hist"], (cv::xfeatures2d::DAISY::NormalizationType)desc_args["norm"], cv::noArray(), true, true);
        }
    }
    else if (descriptor_type == "FREAK")
    {
        if (desc_args.find("patternScale") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::FREAK::create(true, true);
        }
        else if (desc_args.find("nOctaves") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::FREAK::create(true, true, desc_args["patternScale"]);
        }
        else
        {
            descriptor_ = cv::xfeatures2d::FREAK::create(true, true, desc_args["patternScale"], (int)desc_args["nOctaves"]);
        }
    }
    else if (descriptor_type == "LATCH")
    {
        if (desc_args.find("bytes") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::LATCH::create();
        }
        else if (desc_args.find("half_ssd_size") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::LATCH::create((int)desc_args["bytes"]);
        }
        else if (desc_args.find("sigma") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::LATCH::create((int)desc_args["bytes"], true, (int)desc_args["half_ssd_size"]);
        }
        else
        {
            descriptor_ = cv::xfeatures2d::LATCH::create((int)desc_args["bytes"], true, (int)desc_args["half_ssd_size"], desc_args["sigma"]);
        }
    }
    else if (descriptor_type == "LUCID")
    {
        if (desc_args.find("lucid_kernel") == desc_args.end())
        {
            descriptor_= cv::xfeatures2d::LUCID::create();
        }
        else if (desc_args.find("blur_kernel") == desc_args.end())
        {
            descriptor_= cv::xfeatures2d::LUCID::create((int)desc_args["lucid_kernel"]);
        }
        else
        {
            descriptor_= cv::xfeatures2d::LUCID::create((int)desc_args["lucid_kernel"], (int)desc_args["blur_kernel"]);
        }
    }
    else if (descriptor_type == "VGG")
    {
        if (desc_args.find("isigma") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::VGG::create();
        }
        else if (desc_args.find("scale_factor") == desc_args.end())
        {
            descriptor_ = cv::xfeatures2d::VGG::create(cv::xfeatures2d::VGG::VGG_120, desc_args["isigma"]);
        }
        else
        {
            descriptor_ = cv::xfeatures2d::VGG::create(cv::xfeatures2d::VGG::VGG_120, desc_args["isigma"], true, true, desc_args["scale_factor"]);
        }
    }
    else if (descriptor_type == "BOOST")
    {
        float scale = 1.5f;
        if (detector_type == "KAZE" || detector_type == "SURF")
        {
            scale = 6.25f;
        }
        else if (detector_type == "SIFT")
        {
            scale = 6.75f;
        }
        else if (detector_type == "AKAZE" || detector_type == "AGAST" || detector_type == "FAST" || detector_type == "BRISK")
        {
            scale = 5.0f;
        }
        else if (detector_type == "ORB")
        {
            scale = 0.75f;
        }
        descriptor_ = cv::xfeatures2d::BoostDesc::create(cv::xfeatures2d::BoostDesc::BINBOOST_256, true, scale);
    }
}

Detector::Detector(std::string detector_type, std::map<std::string, double> args)
    : Detector(detector_type, std::string(""), args, std::map<std::string, double>())
{
}

Detector::Detector(const Detector &other)
    : Detector(other.detectorType_, other.descriptorType_, other.detectorArgs_, other.descriptorArgs_)
{
}

int Detector::Detect(data::Frame &frame, std::vector<data::Landmark> &landmarks, bool stereo) const
{
    cv::Mat noarr;
    return DetectInRegion(frame, landmarks, noarr, stereo);
}

int Detector::DetectInRectangularRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Point2f start, cv::Point2f end, bool stereo) const
{
    bool compressed = frame.IsCompressed();
    cv::Mat mask = cv::Mat::zeros(frame.GetImage().size(), CV_8U);
    mask(cv::Rect(start, end)) = 255;
    int count = DetectInRegion(frame, landmarks, mask, stereo);
    if (compressed)
    {
        frame.CompressImages();
    }
    return count;
}

int Detector::DetectInRadialRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, double start_r, double end_r, double start_t, double end_t, bool stereo) const
{
    bool compressed = frame.IsCompressed();
    cv::Mat mask = cv::Mat::zeros(frame.GetImage().size(), CV_8U);
    double start_r2 = start_r * start_r;
    double end_r2 = end_r * end_r;
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            double x = j - mask.cols / 2. + 0.5;
            double y = i - mask.rows / 2. + 0.5;
            double r2 = x * x + y * y;
            double t = util::MathUtil::FastAtan2(y, x);
            if (r2 >= start_r2 && r2 < end_r2 && t >= start_t && t < end_t)
            {
                mask.at<unsigned char>(i, j) = 255;
            }
        }
    }
    int count = DetectInRegion(frame, landmarks, mask, stereo);
    if (compressed)
    {
        frame.CompressImages();
    }
    return count;
}

int Detector::DetectInRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Mat &mask, bool stereo) const
{
    bool compressed = frame.IsCompressed();
    std::vector<cv::KeyPoint> kpts;
    const cv::Mat &img = stereo ? frame.GetStereoImage() : frame.GetImage();
    detector_->detect(img, kpts, mask);
    cv::Mat descs;
    if (descriptor_.get() != nullptr)
    {
        if (descriptorType_ == "LUCID")
        {
            cv::Mat rgb;
            cv::cvtColor(img, rgb, cv::COLOR_GRAY2BGR);
            descriptor_->compute(rgb, kpts, descs);
        }
        else
        {
            descriptor_->compute(img, kpts, descs);
        }
    }
    for (int i = 0; i < kpts.size(); i++)
    {
        cv::KeyPoint &kpt = kpts[i];
        data::Landmark landmark;
        if (descriptor_.get() != nullptr)
        {
            cv::Mat desc = descs.row(i);
            data::Feature feat(frame, kpt, desc, stereo);
            if (stereo)
            {
                landmark.AddStereoObservation(feat);
            }
            else
            {
                landmark.AddObservation(feat);
            }
        }
        else
        {
            data::Feature feat(frame, kpt, stereo);
            if (stereo)
            {
                landmark.AddStereoObservation(feat);
            }
            else
            {
                landmark.AddObservation(feat);
            }
        }
        #pragma omp critical
        {
            landmarks.push_back(landmark);
        }
    }
    if (compressed)
    {
        frame.CompressImages();
    }
    return kpts.size();
}

std::string Detector::GetDetectorType()
{
    return detectorType_;
}

std::string Detector::GetDescriptorType()
{
    return descriptorType_;
}

bool Detector::IsDetectorTypeValid(std::string name)
{
    std::vector<std::string> valid = {"GFTT", "FAST", "SIFT", "SURF", "ORB", "BRISK", "STAR", "AKAZE", "KAZE", "AGAST"};
    return std::find(valid.begin(), valid.end(), name) != valid.end();
}

bool Detector::IsDescriptorTypeValid(std::string name)
{
    std::vector<std::string> valid = {"SIFT", "SURF", "ORB", "BRISK", "AKAZE", "KAZE", "DAISY", "FREAK", "LATCH", "LUCID", "VGG", "BOOST"};
    return std::find(valid.begin(), valid.end(), name) != valid.end();
}

bool Detector::IsDetectorDescriptorCombinationValid(std::string det, std::string desc){
    if (!IsDetectorTypeValid(det) || !IsDescriptorTypeValid(desc))
    {
        return false;
    }
    if (desc == "KAZE" && det != "KAZE")
    {
        return false;
    }
    if (desc == "AKAZE" && det != "KAZE" && det != "AKAZE")
    {
        return false;
    }
    if (det == "SIFT" && desc == "ORB")
    {
        return false;
    }
    return true;
}

}
}
