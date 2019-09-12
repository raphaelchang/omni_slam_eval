#ifndef _LAMBDA_TWIST_H_
#define _LAMBDA_TWIST_H_

#include <Eigen/Dense>
#include "util/math_util.h"

using namespace Eigen;

namespace omni_slam
{
namespace odometry
{
namespace LambdaTwist
{

template<typename T>
constexpr T GetNumericLimit()
{
    return 1e-13;
}

template<> constexpr float GetNumericLimit<float>()
{
    // abs limit is 9 digits
    return 1e-7;
}

template<> constexpr double GetNumericLimit<double>()
{
    // abs limit is 17 digits
    return 1e-13;
}

template<> constexpr long double GetNumericLimit<long double>()
{
    // abs limit is 21 digits
    return 1e-15 ;
}

template<typename T, int iterations>
T SolveCubic(T b, T c, T d)
{
    T r0;
    // not monotonic
    if (b * b >= 3.0 * c)
    {
        // h has two stationary points, compute them
        T v = std::sqrt(b * b - 3.0 * c);
        T t1 = (-b - v) / 3.0;

        // Check if h(t1) > 0, in this case make a 2-order approx of h around t1
        T k = ((t1 + b) * t1 + c) * t1 + d;

        if (k > 0.0)
        {
            //Find leftmost root of 0.5*(r0 -t1)^2*(6*t1+2*b) +  k = 0
            r0 = t1 - std::sqrt(-k / (3.0 * t1 + b));
        } else
        {
            T t2 = (-b + v) / (3.0);
            k = ((t2 + b) * t2 + c) * t2 + d;
            //Find rightmost root of 0.5*(r0 -t2)^2*(6*t2+2*b) +  k1 = 0
            r0 = t2 + std::sqrt(-k / (3.0 * t2 + b));
        }
    }
    else
    {
        r0 = -b / 3.0;
        if (std::abs(((T(3.0) * r0 + T(2.0) * b) * r0 + c)) < 1e-4)
        {
            r0 += 1;
        }

    }

    /* Do ITER Newton-Raphson iterations */
    /* Break if position of root changes less than 1e.16 */
    T fx,fpx;
    for (unsigned int cnt = 0; cnt < iterations; ++cnt)
    {
        fx = (((r0 + b) * r0 + c) * r0 + d);

        if ((cnt < 7 || std::abs(fx) > GetNumericLimit<T>()))
        {
            fpx = ((T(3.0) * r0 + T(2.0) * b) * r0 + c);

            r0 -= fx / fpx;
        }
        else
        {
            break;
        }
    }

    return r0;
}

template<typename T>
void SolveEigenWithKnownZero(Matrix<T, 3, 3> x, Matrix<T, 3, 3> &E, Matrix<T, 3, 1> &L)
{
    // one eigenvalue is known to be 0.
    //the known one...
    L(2)=0;

    Matrix<T, 3, 1> v3;
    v3 << x(3) * x(7) - x(6) * x(4),
            x(6) * x(1) - x(7) * x(0),
            x(4) * x(0) - x(3) * x(1);
    v3.normalize();

    T x01_squared = x(0, 1) * x(0, 1);

    // get the two other...
    T b = -x(0, 0) - x(1, 1) - x(2, 2);
    T c = -x01_squared - x(0, 2) * x(0, 2) - x(1, 2) * x(1, 2) +
        x(0, 0) * (x(1, 1) + x(2, 2)) + x(1, 1) * x(2, 2);
    T e1, e2;
    util::MathUtil::Roots(b, c, e1, e2);

    if (std::abs(e1) < std::abs(e2))
    {
        std::swap(e1, e2);
    }
    L(0) = e1;
    L(1) = e2;

    T mx0011 = -x(0, 0) * x(1, 1);
    T prec_0 = x(0, 1) * x(1, 2) - x(0, 2) * x(1, 1);
    T prec_1 = x(0, 1) * x(0, 2) - x(0, 0) * x(1, 2);

    T e = e1;
    T tmp = 1.0 / (e *(x(0, 0) + x(1, 1)) + mx0011 - e * e + x01_squared);
    T a1 = -(e * x(0, 2) + prec_0) * tmp;
    T a2 = -(e * x(1, 2) + prec_1) * tmp;
    T rnorm = ((T)1.0) / std::sqrt(a1 * a1 + a2 * a2 + 1.0);
    a1 *= rnorm;
    a2 *= rnorm;
    Matrix<T, 3, 1> v1(a1, a2, rnorm);

    T tmp2 = 1.0 / (e2 * (x(0, 0) + x(1, 1)) + mx0011 - e2 * e2 + x01_squared);
    T a21 = -(e2 * x(0, 2) + prec_0) * tmp2;
    T a22 = -(e2 * x(1, 2) + prec_1) * tmp2;
    T rnorm2 = 1.0 / std::sqrt(a21 * a21 + a22 * a22 + 1.0);
    a21 *= rnorm2;
    a22 *= rnorm2;
    Matrix<T, 3, 1> v2(a21, a22, rnorm2);

    E = Matrix<T,3,3>(v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], v1[2], v2[2], v3[2]);
}

template<typename T, int iterations>
void GaussNewtonRefineL(Matrix<T, 3, 1> &L, T a12, T a13, T a23, T b12, T b13, T b23 ){

    for(int i=0; i < iterations; ++i)
    {
        T l1 = L(0);
        T l2 = L(1);
        T l3 = L(2);
        T r1 = l1 * l1 + l2 * l2 + b12 * l1 * l2 - a12;
        T r2 = l1 * l1 + l3 * l3 + b13 * l1 * l3 - a13;
        T r3 = l2 * l2 + l3 * l3 + b23 * l2 * l3 - a23;

        if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-10)
        {
            break;
        }

        T dr1dl1 = 2.0 * l1 + b12 * l2;
        T dr1dl2 = 2.0 * l2 + b12 * l1;
        T dr2dl1 = 2.0 * l1 + b13 * l3;
        T dr2dl3 = 2.0 * l3 + b13 * l1;
        T dr3dl2 = 2.0 * l2 + b23 * l3;
        T dr3dl3 = 2.0 * l3 + b23 * l2;

        Matrix<T, 3, 1> r;
        r << r1, r2, r3;

        {
            T v0 = dr1dl1;
            T v1 = dr1dl2;
            T v3 = dr2dl1;
            T v5 = dr2dl3;
            T v7 = dr3dl2;
            T v8 = dr3dl3;
            T det = 1.0 / (-v0 * v5 * v7 - v1 * v3 * v8);

            Matrix<T, 3, 3> Ji;
            Ji << -v5 * v7, -v1 * v8,  v1 * v5,
                    -v3 * v8,  v0 * v8, -v0 * v5,
                    v3 * v7, -v0 * v7, -v1 * v3;
            Matrix<T, 3, 1> L1 = L - det * (Ji * r);

            {
                T l1 = L1(0);
                T l2 = L1(1);
                T l3 = L1(2);
                T r11 = l1 * l1 + l2 * l2 + b12 * l1 * l2 - a12;
                T r12 = l1 * l1 + l3 * l3 + b13 * l1 * l3 - a13;
                T r13 = l2 * l2 + l3 * l3 + b23 * l2 * l3 - a23;
                if(std::abs(r11) + std::abs(r12) + std::abs(r13) > std::abs(r1) + std::abs(r2) + std::abs(r3))
                {
                    break;
                }
                else
                {
                    L = L1;
                }
            }
        }
    }
}

template<typename T, int refinement_iterations = 5>
int P3P( Matrix<T, 3, 1> y1, Matrix<T, 3, 1> y2, Matrix<T, 3, 1> y3, Matrix<T, 3, 1> x1, Matrix<T, 3, 1> x2, Matrix<T, 3, 1> x3, std::vector<Matrix<T, 3, 3>>& Rs, std::vector<Matrix<T, 3, 1>>& Ts)
{
    y1.normalize();
    y2.normalize();
    y3.normalize();


    T b12 = -2.0 * (y1.dot(y2));
    T b13 = -2.0 * (y1.dot(y3));
    T b23 = -2.0 * (y2.dot(y3));

    Matrix<T, 3, 1> d12 = x1 - x2;
    Matrix<T, 3, 1> d13 = x1 - x3;
    Matrix<T, 3, 1> d23 = x2 - x3;
    Matrix<T, 3, 1> d12xd13 = d12.cross(d13);

    T a12 = d12.squaredNorm();
    T a13 = d13.squaredNorm();
    T a23 = d23.squaredNorm();

    T c31 = -0.5 * b13;
    T c23 = -0.5 * b23;
    T c12 = -0.5 * b12;
    T blob = c12 * c23 * c31 - 1.0;

    T s31_squared = 1.0 - c31 * c31;
    T s23_squared = 1.0 - c23 * c23;
    T s12_squared = 1.0 - c12 * c12;

    T p3 = (a13 * (a23 * s31_squared - a13 * s23_squared));

    T p2 = 2.0 * blob * a23 * a13 + a13 * (2.0 * a12 + a13) * s23_squared + a23 * (a23 - a12) * s31_squared;

    T p1 = a23 * (a13 - a23) * s12_squared - a12 * a12 * s23_squared - 2.0 * a12 * (blob * a23 + a13 * s23_squared);

    T p0 = a12 * (a12 * s23_squared - a23 * s12_squared);

    T g=0;

    p3 = 1.0 / p3;
    p2 *= p3;
    p1 *= p3;
    p0 *= p3;

    // get sharpest real root of above...
    g = SolveCubic(p2, p1, p0);

    T A00 = a23 * (1.0- g);
    T A01 = (a23 * b12) * 0.5;
    T A02 = (a23 * b13 * g) * -0.5;
    T A11 = a23 - a12 + a13 * g;
    T A12 = b23 * (a13 * g - a12) * 0.5;
    T A22 = g * (a13 - a23) - a12;

    Matrix<T, 3, 3> A;
    A << A00, A01, A02,
        A01, A11, A12,
        A02, A12, A22;

    // get sorted eigenvalues and eigenvectors given that one should be zero...
    Matrix<T, 3, 3> V;
    Matrix<T, 3, 1> L;

    SolveEigenWithKnownZero(A, V, L);

    T v = std::sqrt(std::max(T(0), -L(1) / L(0)));

    int valid = 0;
    std::vector<Matrix<T, 3, 1>> Ls(4);

    {
        T s = v;

        T w2 = T(1.0) / (s * V(1) - V(0));
        T w0 = (V(3) - s * V(4)) * w2;
        T w1 = (V(6) - s * V(7)) * w2;

        T a = T(1.0) / ((a13 - a12) * w1 * w1 - a12 * b13 * w1 - a12);
        T b = (a13 * b12 * w1 - a12 * b13 * w0 - T(2.0) * w0 * w1 * (a12 - a13)) * a;
        T c = ((a13 - a12) * w0 * w0 + a13 * b12 * w0 + a13) * a;

        if(b * b - 4.0 * c >= 0)
        {
            T tau1, tau2;
            util::MathUtil::Roots(b, c, tau1, tau2);
            if(tau1 > 0)
            {
                T tau = tau1;
                T d = a23 / (tau * (b23 + tau) + T(1.0));
                T l2 = std::sqrt(d);
                T l3 = tau * l2;

                T l1 = w0 * l2 + w1 * l3;
                if(l1 >= 0)
                {
                    Ls[valid] << l1, l2, l3;
                    ++valid;
                }
            }
            if(tau2 > 0)
            {
                T tau = tau2;
                T d = a23 / (tau * (b23 + tau) + T(1.0));
                T l2 = std::sqrt(d);
                T l3 = tau * l2;
                T l1 = w0 * l2 + w1 * l3;
                if(l1 >= 0)
                {
                    Ls[valid] << l1, l2, l3;
                    ++valid;
                }
            }
        }
    }

    {
        T s=-v;
        T w2 = T(1.0) / ( s * V(0, 1) - V(0, 0));
        T w0 = (V(1, 0) - s * V(1, 1)) * w2;
        T w1 = (V(2, 0) - s * V(2, 1)) * w2;

        T a = T(1.0) / ((a13 - a12) * w1 * w1 - a12 * b13 * w1 - a12);
        T b = (a13 * b12 * w1 - a12 * b13 * w0 - T(2.0) * w0 * w1 * (a12 - a13)) * a;
        T c = ((a13 - a12) * w0 * w0 + a13 * b12 * w0 + a13) * a;

        if(b * b - 4.0 * c >= 0){
            T tau1, tau2;

            util::MathUtil::Roots(b, c, tau1, tau2);
            if(tau1 > 0)
            {
                T tau = tau1;
                T d = a23 / (tau * (b23 + tau) + T(1.0));
                if(d > 0)
                {
                    T l2 = std::sqrt(d);

                    T l3 = tau * l2;

                    T l1 = w0 * l2 + w1 * l3;
                    if(l1 >= 0)
                    {
                        Ls[valid] << l1, l2, l3;
                        ++valid;
                    }
                }
            }
            if (tau2 > 0){
                T tau = tau2;
                T d = a23 / (tau * (b23 + tau) + T(1.0));
                if (d > 0)
                {
                    T l2 = std::sqrt(d);

                    T l3 = tau * l2;

                    T l1 = w0 * l2 + w1 * l3;
                    if (l1 >= 0)
                    {
                        Ls[valid] << l1, l2, l3;
                        ++valid;
                    }
                }
            }
        }
    }

    for(int i = 0; i < valid; ++i)
    {
        GaussNewtonRefineL<T, refinement_iterations>(Ls[i], a12, a13, a23, b12, b13, b23);
    }

    Matrix<T, 3, 1> ry1, ry2, ry3;
    Matrix<T, 3, 1> yd1;
    Matrix<T, 3, 1> yd2;
    Matrix<T, 3, 1> yd1xd2;
    Matrix<T, 3, 3> X;
    X << d12(0),d13(0),d12xd13(0),
        d12(1),d13(1),d12xd13(1),
        d12(2),d13(2),d12xd13(2);
    X=X.inverse();

    Rs.clear();
    Rs.resize(4);
    Ts.clear();
    Ts.resize(4);
    for(int i = 0; i < valid; ++i)
    {
        ry1 = y1 * Ls(i)(0);
        ry2 = y2 * Ls(i)(1);
        ry3 = y3 * Ls(i)(2);

        yd1 = ry1 - ry2;
        yd2 = ry1 - ry3;
        yd1xd2 = yd1.cross(yd2);

        Matrix<T, 3, 3> Y;
        Y << yd1(0), yd2(0), yd1xd2(0),
            yd1(1), yd2(1), yd1xd2(1),
            yd1(2), yd2(2), yd1xd2(2);

        Rs[i] = Y * X;
        Ts[i] = (ry1 - Rs[i] * x1);
    }

    return valid;
}

}
}
}

#endif /* _LAMBDA_TWIST_H_ */
