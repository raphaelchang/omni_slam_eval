#ifndef _REPROJECTION_ERROR_H_
#define _REPROJECTION_ERROR_H_

#include "camera/camera_model.h"
#include "camera/double_sphere.h"
#include "data/feature.h"
#include <ceres/ceres.h>
#include "util/tf_util.h"

namespace omni_slam
{
namespace optimization
{

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
        //const camera::CameraModel<T> &camera((camera::CameraModel<T>&)feature_.GetFrame().GetCameraModel());
        const camera::DoubleSphere<T> camera(T(295.936), T(295.936), T(512), T(512), T(0.3), T(0.6666667));
        const Matrix<T, 3, 4> pose = Map<const Matrix<T, 3, 4>>(camera_pose);
        const Matrix<T, 3, 1> worldPt = Map<const Matrix<T, 3, 1>>(point);
        const Matrix<T, 3, 1> camPt = util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(pose, worldPt));
        if (camera.ProjectToImage(camPt, reprojPoint))
        {
            reproj_error[0] = reprojPoint(0) - T(feature_.GetKeypoint().pt.x);
            reproj_error[1] = reprojPoint(1) - T(feature_.GetKeypoint().pt.y);
            return true;
        }
        reproj_error[0] = T(0.);
        reproj_error[1] = T(0.);
        return true;
    }

    static ceres::CostFunction* Create(const data::Feature &feature)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 12, 3>(new ReprojectionError(feature));
    }

private:
    const data::Feature feature_;
};

}
}

#endif /* _REPROJECTION_ERROR_H_ */
