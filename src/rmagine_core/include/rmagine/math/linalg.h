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
#include <rmagine/types/shared_functions.h>

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
RMAGINE_FUNCTION
void svd(
    const Matrix3x3& A,
    Matrix3x3& U,
    Matrix3x3& W,
    Matrix3x3& V
);

RMAGINE_FUNCTION
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
RMAGINE_INLINE_FUNCTION
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
        ret.R.set(U * S * V.transpose());
        ret.R.normalizeInplace();
        ret.t = m - ret.R * d;
    } else {
        ret.setIdentity();
    }

    return ret;
}



} // namespace rmagine

#include "linalg.tcc"

#endif // RMAGINE_MATH_MATH_LINALG_H