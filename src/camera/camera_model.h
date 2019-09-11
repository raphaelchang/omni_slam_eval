#ifndef _CAMERA_MODEL_H_
#define _CAMERA_MODEL_H_

#include <string>
#include <Eigen/Dense>

using namespace Eigen;

namespace omni_slam
{
namespace camera
{

template <typename T = double>
class CameraModel
{
public:
    CameraModel(std::string name)
    {
        name_ = name;
    }
    virtual bool ProjectToImage(const Matrix<T, 3, 1> &bearing, Matrix<T, 2, 1> &pixel) const = 0;
    virtual bool UnprojectToBearing(const Matrix<T, 2, 1> &pixel, Matrix<T, 3, 1> &bearing) const = 0;
    virtual T GetFOV() = 0;

private:
    std::string name_;
};

}
}

#endif /* _CAMERA_MODEL_H_ */
