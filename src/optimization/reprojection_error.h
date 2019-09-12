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
        : feature_(feature)
    {
    }

    template<typename T>
    bool operator()(const T* const camera_pose, const T* const point, T *reproj_error) const
    {
        Matrix<T, 2, 1> reprojPoint;
        C<T> camera(feature_.GetFrame().GetCameraModel());
        const Matrix<T, 3, 4> pose = Map<const Matrix<T, 3, 4>>(camera_pose);
        const Matrix<T, 3, 1> worldPt = Map<const Matrix<T, 3, 1>>(point);
        const Matrix<T, 3, 1> camPt = util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(pose, worldPt));
        camera.ProjectToImage(camPt, reprojPoint);
        reproj_error[0] = reprojPoint(0) - T(feature_.GetKeypoint().pt.x);
        reproj_error[1] = reprojPoint(1) - T(feature_.GetKeypoint().pt.y);
        return true;
    }

    static ceres::CostFunction* Create(const data::Feature &feature)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 12, 3>(new ReprojectionError<C>(feature));
    }

private:
    const data::Feature feature_;
};

}
}

#endif /* _REPROJECTION_ERROR_H_ */
