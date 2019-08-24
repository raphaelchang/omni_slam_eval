#ifndef _CAMERA_MODEL_H_
#define _CAMERA_MODEL_H_

#include <string>
#include <Eigen/Dense>

using namespace Eigen;

namespace omni_slam
{
namespace camera
{

class CameraModel
{
public:
    CameraModel(std::string name)
    {
        name_ = name;
    }
    virtual bool ProjectToImage(const Vector3d &bearing, Vector2d &pixel) const = 0;
    virtual bool UnprojectToBearing(const Vector2d &pixel, Vector3d &bearing) const = 0;
    virtual double GetFOV() = 0;

private:
    std::string name_;
};

}
}

#endif /* _CAMERA_MODEL_H_ */
