/*
 * Copyright (c) 2022, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * 
 * @brief Linear Algebra Function
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_LINALG_H
#define RMAGINE_MATH_LINALG_H

#include "types.h"
#include "math.h"
#include <rmagine/types/shared_functions.h>
#include <rmagine/types/PointCloud.hpp>

#include <rmagine/util/prints.h>

namespace rmagine
{

/**
 * @brief Composes Rotational, translational and scale parts 
 * into one 4x4 affine transformation matrix
 * 
 * @param T   transform object consisting of translational and rotational parts
 * @param scale   scale vector
 * @return Matrix4x4   composed 4x4 transformation matrix
 */
RMAGINE_FUNCTION
Matrix4x4 compose(const Transform& T, const Vector3& scale);

RMAGINE_FUNCTION
Matrix4x4 compose(const Transform& T, const Matrix3x3& S);

/**
 * @brief Decomposes Affine Transformation Matrix 
 * (including Scale) into its Translation, Rotation and Scale components
 * 
 * @param M 4x4 affine transformation matrix
 * @param T transform object consisting of translational and rotational parts
 * @param S 3x3 scale matrix
 */
RMAGINE_FUNCTION
void decompose(const Matrix4x4& M, Transform& T, Matrix3x3& S);

/**
 * @brief Decomposes Affine Transformation Matrix 
 * (including Scale) into its Translation, Rotation and Scale components
 * 
 * @param M 4x4 affine transformation matrix
 * @param T transform object consisting of translational and rotational parts
 * @param scale scale vector
 */
RMAGINE_FUNCTION
void decompose(const Matrix4x4& M, Transform& T, Vector3& scale);

/**
 * @brief linear inter- or extrapolate between A and B with a factor
 * 
 * Examples:
 * - if fac=0.0 it is exactly A
 * - if fac=0.5 it is exactly between A and B
 * - if fac=1.0 it is exactly B
 * - if fac=2.0 it it goes on a (B-A) length from B (extrapolation)
*/
RMAGINE_FUNCTION
Quaternion polate(const Quaternion& A, const Quaternion& B, float fac);

/**
 * @brief linear inter- or extrapolate between A and B with a factor
 * 
 * Examples:
 * - if fac=0.0 it is exactly A
 * - if fac=0.5 it is exactly between A and B
 * - if fac=1.0 it is exactly B
 * - if fac=2.0 it it goes on a (B-A) length from B (extrapolation)
*/
RMAGINE_FUNCTION
Transform polate(const Transform& A, const Transform& B, float fac);


/**
 * @brief Cholesky Decomposition L*L^T = A
 * 
 * A needs to be quadric, positive-definit, and symmetric!
 * (For performance reasons "positive-definit, and symmetric" is not checked. It is planned to add a check at least for debug compilation)
 * 
 */
template<typename DataT, unsigned int Dim>
void chol(
  const Matrix_<DataT, Dim, Dim>& A,
  Matrix_<DataT, Dim, Dim>& L
);

// Numerical Recipes
// M = MatrixT::rows()
// N = MatrixT::cols()
// 
// Warning: Numerical Recipes has different SVD matrix shapes
// than Wikipedia
template<typename MatrixT>
struct svd_dims {
    using U = MatrixT; // same as input
    using w = Matrix_<typename MatrixT::Type, MatrixT::cols(), 1>;
    using W = Matrix_<typename MatrixT::Type, MatrixT::cols(), MatrixT::cols()>;
    using V = Matrix_<typename MatrixT::Type, MatrixT::cols(), MatrixT::cols()>;
};

/**
 * @brief own SVD implementation. 
 * Why use it? 
 * - ~2x faster than Eigen
 * - SOON: Works insided of CUDA kernels
 *
 */
template<typename DataT, unsigned int Rows, unsigned int Cols>
void svd(
    const Matrix_<DataT, Rows, Cols>& A, 
    Matrix_<DataT, Rows, Cols>& U,
    Matrix_<DataT, Cols, Cols>& W, // matrix
    Matrix_<DataT, Cols, Cols>& V
);

template<typename DataT, unsigned int Rows, unsigned int Cols>
void svd(
    const Matrix_<DataT, Rows, Cols>& A,
    Matrix_<DataT, Rows, Cols>& U,
    Matrix_<DataT, Cols, 1>& w, // vector version (Cols should be something with max)
    Matrix_<DataT, Cols, Cols>& V
);

/**
 * @brief SVD that can be used for both CPU and GPU (Cuda kernels)
 *
 */
RMAGINE_DEVICE_FUNCTION
void svd(
    const Matrix3x3& A,
    Matrix3x3& U,
    Matrix3x3& W,
    Matrix3x3& V
);

RMAGINE_DEVICE_FUNCTION
void svd(
    const Matrix3x3& A, 
    Matrix3x3& U,
    Vector3& w,
    Matrix3x3& V
);

/**
 * @brief computes the optimal transformation according to Umeyama's algorithm 
 * 
 * Note: sometimes referred to as Kabsch/Umeyama
 * 
 * @param n_meas: if == 0: Resulting Transform is set to identity. Otherwise the standard Umeyama algorithm is performed
 * 
 */
RMAGINE_INLINE_DEVICE_FUNCTION
Transform umeyama_transform(
    const Vector3& d,
    const Vector3& m,
    const Matrix3x3& C,
    const unsigned int n_meas = 1
)
{
    Transform ret;

    if(n_meas > 0)
    {
        // intermediate storage needed (yet)
        Matrix3x3 U, S, V;
        svd(C, U, S, V);
        S.setIdentity();
        if(U.det() * V.det() < 0)
        {
            S(2, 2) = -1;
        }
        ret.R.set(U * S * V.transpose());
        ret.R.normalizeInplace();
        ret.t = m - ret.R * d;
    } else {
        ret.setIdentity();
    }

    return ret;
}

/**
 * @brief computes the optimal transformation according to Umeyama's algorithm 
 * 
 * Note: sometimes referred to as Kabsch/Umeyama
 * 
 * @param n_meas: if == 0: Resulting Transform is set to identity. Otherwise the standard Umeyama algorithm is performed
 * 
 */
RMAGINE_INLINE_DEVICE_FUNCTION
Transform umeyama_transform(
    const CrossStatistics& stats)
{
    Transform ret;

    if(stats.n_meas > 0)
    {
        // intermediate storage needed (yet)
        Matrix3x3 U, S, V;
        svd(stats.covariance, U, S, V);

        // std::cout << "BLAAAA" << std::endl;

        S.setIdentity();
        if(U.det() * V.det() < 0)
        {
            S(2, 2) = -1;
        }
        ret.R.set(U * S * V.transpose());
        ret.R.normalizeInplace();
        ret.t = stats.model_mean - ret.R * stats.dataset_mean;
    } else {
        ret.setIdentity();
    }

    return ret;
}

RMAGINE_INLINE_FUNCTION
Matrix3x3 so3_hat(Vector3f v)
{
    Matrix3x3 M;
    M(0,0) =  0.0; M(1,0) = -v.z; M(2,0) =  v.y;
    M(0,1) =  v.z; M(1,1) =  0.0; M(2,1) = -v.x;
    M(0,2) = -v.y; M(1,2) =  v.x; M(2,2) =  0.0;
    return M;
}

RMAGINE_INLINE_FUNCTION
float GM_weight(
    const float kernel_scale,
    const float residual2)
{
    return sqr(kernel_scale) / sqr(kernel_scale + residual2);
}

// TODO: test 
RMAGINE_INLINE_FUNCTION
void jacobian_and_residual_p2p(
    Matrix_<float, 3, 6>& Jr, // [out] Jacobian 
    Matrix_<float, 3, 1>& residual, // [out] residual
    const Vector3f& Pm, // model point
    const Vector3f& Pd) // dataset point
{
    // this is slightly different from kiss icp
    // TODO: test

    // Jr top 3x3 = Identity
    Jr(0,0) = 1.0; Jr(1,0) = 0.0; Jr(2,0) = 0.0;
    Jr(0,1) = 0.0; Jr(1,1) = 1.0; Jr(2,1) = 0.0;
    Jr(0,2) = 0.0; Jr(1,2) = 0.0; Jr(2,2) = 1.0;

    // Jr bottom 3x3 = -1 * hat(Pd) -> only hat
    Jr(0,3) =   0.0; Jr(1,3) = -Pd.z; Jr(2,3) =  Pd.y;
    Jr(0,4) =  Pd.z; Jr(1,4) =   0.0; Jr(2,4) = -Pd.x;
    Jr(0,5) = -Pd.y; Jr(1,5) =  Pd.x; Jr(2,5) =   0.0;

    const Vector3f res_vec = Pm - Pd; // Pm <- Pd
    residual(0,0) = res_vec.x;
    residual(1,0) = res_vec.y;
    residual(2,0) = res_vec.z;
}


RMAGINE_INLINE_FUNCTION
void jacobian_and_residual_p2l(
    Matrix_<float, 1, 6>& J, // [out] Jacobian
    float& residual, // [out] residual
    const Vector3f& Pm, // model point 
    const Vector3f& Nm, // model normal
    const Vector3f& Pd) // dataset point
{
    // this is slightly different from kiss icp
    // TODO: test

    // TODO: test if we should better write Pm - Pd
    residual = (Pd - Pm).dot(Nm);
    // put this outside?
    // const float w = GM_weight(5.0, residual);

    const Vector3f PdNm = Pd.cross(Nm);

    J(0,0) = PdNm.x;
    J(0,1) = PdNm.y;
    J(0,2) = PdNm.z;
    J(0,3) = Nm.x;
    J(0,4) = Nm.y;
    J(0,5) = Nm.z;
}


/**
 * @brief Build a Gauss-Newton linear system of the form 
 * (J^T * W * J) * x = (J^T * W * r)
 * using point to point metric (P2P)
 */
RMAGINE_INLINE_FUNCTION
void build_linear_system_p2p(
    Matrix_<float, 6, 6>& JTwJ,
    Matrix_<float, 6, 1>& JTwr,
    const MemoryView<Vector, RAM>& model_points, 
    const MemoryView<Vector, RAM>& dataset_points)
{
    // TODO:
    // - test
    // - make reduction from this
    for(size_t i=0; i<model_points.size(); i++)
    {
        Matrix_<float, 3, 6> J;
        Matrix_<float, 3, 1> r;
        
        jacobian_and_residual_p2p(J, r, 
          model_points[i], dataset_points[i]);

        float residual2 = sqr(r(0,0)) + sqr(r(1,0)) + sqr(r(2,0)); 
        float w = GM_weight(5.0, residual2);
        JTwJ += (J.T() * w) * J; 
        JTwr += (J.T() * w) * r;
    }
}

/**
 * @brief Build a Gauss-Newton linear system of the form 
 * (J^T * W * J) * x = (J^T * W * r)
 * using point to plane metric (P2L)
 */
RMAGINE_INLINE_FUNCTION
void build_linear_system_p2l(
    Matrix_<float, 6, 6>& JTwJ,
    Matrix_<float, 6, 1>& JTwr,
    const MemoryView<Vector, RAM>& model_points,
    const MemoryView<Vector, RAM>& model_normals, 
    const MemoryView<Vector, RAM>& dataset_points)
{
    // TODO:
    // - test
    // - make reduction from this
    for(size_t i=0; i<model_points.size(); i++)
    {
        Matrix_<float, 1, 6> J;
        float r;
        jacobian_and_residual_p2l(J, r, 
          model_points[i], model_normals[i], dataset_points[i]);

        float weight = GM_weight(5.0, r);
        JTwJ += (J.T() * weight) * J; 
        JTwr += (J.T() * weight) * r;
    }
}

RMAGINE_INLINE_FUNCTION
void build_linear_system_p2p(
    Matrix_<float, 6, 6>& JTwJ,
    Matrix_<float, 6, 1>& JTwr,
    const PointCloudView_<RAM>& model, 
    const PointCloudView_<RAM>& dataset)
{   
    // TODO:
    // - test
    // - make reduction from this
    for(size_t i=0; i<model.points.size(); i++)
    {
        Matrix_<float, 3, 6> J;
        Matrix_<float, 3, 1> r;
        
        jacobian_and_residual_p2p(J, r, 
          model.points[i], dataset.points[i]);

        float residual2 = sqr(r(0,0)) + sqr(r(1,0)) + sqr(r(2,0)); 
        float w = GM_weight(5.0, residual2);
        JTwJ += (J.T() * w) * J; 
        JTwr += (J.T() * w) * r;
    }
}

RMAGINE_INLINE_FUNCTION
void build_linear_system_p2l(
    Matrix_<float, 6, 6>& JTwJ,
    Matrix_<float, 6, 1>& JTwr,
    const PointCloudView_<RAM>& model,
    const PointCloudView_<RAM>& dataset)
{   
    // TODO:
    // - test
    // - make reduction from this
    for(size_t i=0; i<model.points.size(); i++)
    {
        Matrix_<float, 1, 6> J;
        float r;
        jacobian_and_residual_p2l(J, r, 
          model.points[i], model.normals[i], dataset.points[i]);

        float weight = GM_weight(5.0, r);
        JTwJ += (J.T() * weight) * J; 
        JTwr += (J.T() * weight) * r;
    }
}

// Collection of minimization strategies
//
// Umeyama
// 
// https://github.com/pglira/simpleICP
// P2L minimization
// (Pd x Nm) * {rx,ry,rz} + Nm * {tx,ty,tz} = Nm * (Pm - Pd)
// -> A*x = b
// 

} // namespace rmagine

#include "linalg.tcc"

#endif // RMAGINE_MATH_MATH_LINALG_H