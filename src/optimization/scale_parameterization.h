#ifndef _SCALE_PARAMETERIZATION_H_
#define _SCALE_PARAMETERIZATION_H_

#include <ceres/ceres.h>
#include <Eigen/Dense>

using namespace Eigen;

namespace omni_slam
{
namespace optimization
{

template<int G>
class ScaleParameterization
{
public:
    ScaleParameterization(const Matrix<double, G, 1> &vec)
        : vec_(vec.normalized())
    {
    }

    template<typename T>
    bool operator()(const T *x, const T *delta, T *x_plus_delta) const
    {
        Matrix<T, G, 1> vec = vec_.template cast<T>();
        for (int i = 0; i < G; i++)
        {
            x_plus_delta[i] = x[i] + delta[0] * vec[i];
        }
        return true;
    }

    static ceres::LocalParameterization* Create(const Matrix<double, G, 1> &vec)
    {
        return new ceres::AutoDiffLocalParameterization<ScaleParameterization, G, 1>(new ScaleParameterization<G>(vec));
    }

private:
    const Matrix<double, G, 1> vec_;
};

}
}

#endif /* _SCALE_PARAMETERIZATION_H_ */
