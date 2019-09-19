#ifndef _REPROJECTION_ERROR_H_
#define _REPROJECTION_ERROR_H_

#include "data/feature.h"
#include <ceres/ceres.h>
#include "util/tf_util.h"

namespace omni_slam
{
namespace optimization
{

template <template<typename> class C>
class ReprojectionError
{
public:
    ReprojectionError(const data::Feature &feature)
        : feature_(feature),
        stereoFeature_(feature),
        hasStereo_(false)
    {
    }

    ReprojectionError(const data::Feature &feature, const data::Feature &stereo_feature)
        : feature_(feature),
        stereoFeature_(stereo_feature),
        hasStereo_(true)
    {
    }

    template<typename T>
    bool operator()(const T* const camera_orientation, const T* const camera_translation, const T* const point, T *reproj_error) const
    {
        Matrix<T, 2, 1> reprojPoint;
        C<T> camera(feature_.GetFrame().GetCameraModel());
        const Quaternion<T> orientation = Map<const Quaternion<T>>(camera_orientation);
        const Matrix<T, 3, 1> translation = Map<const Matrix<T, 3, 1>>(camera_translation);
        const Matrix<T, 3, 4> pose = util::TFUtil::QuaternionTranslationToPoseMatrix(orientation, translation);
        const Matrix<T, 3, 1> worldPt = Map<const Matrix<T, 3, 1>>(point);
        const Matrix<T, 3, 1> camPt = util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(pose, worldPt));
        camera.ProjectToImage(camPt, reprojPoint);
        reproj_error[0] = reprojPoint(0) - T(feature_.GetKeypoint().pt.x);
        reproj_error[1] = reprojPoint(1) - T(feature_.GetKeypoint().pt.y);
        if (hasStereo_ && stereoFeature_.GetFrame().HasStereoImage())
        {
            Matrix<T, 2, 1> reprojPoint2;
            Matrix<T, 3, 4> stereoPose = stereoFeature_.GetFrame().GetStereoPose().cast<T>();
            camera.ProjectToImage(util::TFUtil::TransformPoint(stereoPose, camPt), reprojPoint2);
            reproj_error[0] += reprojPoint(0) - T(stereoFeature_.GetKeypoint().pt.x);
            reproj_error[1] += reprojPoint(1) - T(stereoFeature_.GetKeypoint().pt.y);
        }
        return true;
    }

    static ceres::CostFunction* Create(const data::Feature &feature)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(new ReprojectionError<C>(feature));
    }

    static ceres::CostFunction* Create(const data::Feature &feature, const data::Feature &stereo_feature)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(new ReprojectionError<C>(feature, stereo_feature));
    }

private:
    const data::Feature feature_;
    const data::Feature stereoFeature_;
    bool hasStereo_;
};

}
}

#endif /* _REPROJECTION_ERROR_H_ */
