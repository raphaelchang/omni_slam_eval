#ifndef _DISTORTION_MODEL_H_
#define _DISTORTION_MODEL_H_

#include <string>
#include <Eigen/Dense>

using namespace Eigen;

namespace omni_slam
{
namespace camera
{

template <typename T = double>
class DistortionModel
{
public:
    DistortionModel(std::string name)
    {
        name_ = name;
    }
    virtual bool Undistort(const Matrix<T, 2, 1> &pixel_dist, Matrix<T, 2, 1> &pixel_undist) const = 0;
    virtual bool Distort(const Matrix<T, 2, 1> &pixel_undist, Matrix<T, 2, 1> &pixel_dist) const = 0;

private:
    std::string name_;
};

}
}

#endif /* _DISTORTION_MODEL_H_ */
