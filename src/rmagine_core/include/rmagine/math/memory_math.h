/*
 * Copyright (c) 2025, University Osnabrück
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
 * @brief Contains functions to do math on many elements stored in a rmagine::Memory object
 *
 * @date 13.08.2025
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2025, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MATH_MEMORY_MATH_H
#define RMAGINE_MATH_MEMORY_MATH_H

#include <rmagine/types/Memory.hpp>
#include <rmagine/math/types.h>

namespace rmagine
{

/////////////
// #multNxN
////////
void multNxN(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Quaternion, RAM>& B,
    MemoryView<Quaternion, RAM>& C);

Memory<Quaternion, RAM> multNxN(
    const MemoryView<Quaternion, RAM>& A, 
    const MemoryView<Quaternion, RAM>& B);

void multNxN(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Vector, RAM>& b, 
    MemoryView<Vector, RAM>& c);

Memory<Vector, RAM> multNxN(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Vector, RAM>& b);

void multNxN(
    const MemoryView<Transform, RAM>& T1,
    const MemoryView<Transform, RAM>& T2,
    MemoryView<Transform, RAM>& Tr);

Memory<Transform, RAM> multNxN(
    const MemoryView<Transform, RAM>& T1,
    const MemoryView<Transform, RAM>& T2);

void multNxN(
    const MemoryView<Transform, RAM>& T,
    const MemoryView<Vector, RAM>& x,
    MemoryView<Vector, RAM>& c);

Memory<Vector, RAM> multNxN(
    const MemoryView<Transform, RAM>& T,
    const MemoryView<Vector, RAM>& x);

void multNxN(
    const MemoryView<Matrix3x3, RAM>& M1,
    const MemoryView<Matrix3x3, RAM>& M2,
    MemoryView<Matrix3x3, RAM>& Mr);

Memory<Matrix3x3, RAM> multNxN(
    const MemoryView<Matrix3x3, RAM>& M1,
    const MemoryView<Matrix3x3, RAM>& M2);

void multNxN(
    const MemoryView<Matrix3x3, RAM>& M,
    const MemoryView<Vector, RAM>& x,
    MemoryView<Vector, RAM>& c);

Memory<Vector, RAM> multNxN(
    const MemoryView<Matrix3x3, RAM>& M,
    const MemoryView<Vector, RAM>& x);

/////////////
// #multNx1
////////
void multNx1(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Quaternion, RAM>& b,
    MemoryView<Quaternion, RAM>& C);

Memory<Quaternion, RAM> multNx1(
    const MemoryView<Quaternion, RAM>& A, 
    const MemoryView<Quaternion, RAM>& b);

void multNx1(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Vector, RAM>& b, 
    MemoryView<Vector, RAM>& C);

Memory<Vector, RAM> multNx1(
    const MemoryView<Quaternion, RAM>& A,
    const MemoryView<Vector, RAM>& b);

void multNx1(
    const MemoryView<Transform, RAM>& T1,
    const MemoryView<Transform, RAM>& t2,
    MemoryView<Transform, RAM>& Tr);

Memory<Transform, RAM> multNx1(
    const MemoryView<Transform, RAM>& T1,
    const MemoryView<Transform, RAM>& t2);

void multNx1(
    const MemoryView<Transform, RAM>& T,
    const MemoryView<Vector, RAM>& x,
    MemoryView<Vector, RAM>& C);

Memory<Vector, RAM> multNx1(
    const MemoryView<Transform, RAM>& T,
    const MemoryView<Vector, RAM>& x);

void multNx1(
    const MemoryView<Matrix3x3, RAM>& M1,
    const MemoryView<Matrix3x3, RAM>& m2,
    MemoryView<Matrix3x3, RAM>& Mr);

Memory<Matrix3x3, RAM> multNx1(
    const MemoryView<Matrix3x3, RAM>& M1,
    const MemoryView<Matrix3x3, RAM>& m2);

void multNx1(
    const MemoryView<Matrix3x3, RAM>& M,
    const MemoryView<Vector, RAM>& x,
    MemoryView<Vector, RAM>& C);

Memory<Vector, RAM> multNx1(
    const MemoryView<Matrix3x3, RAM>& M,
    const MemoryView<Vector, RAM>& x);

/////////////
// #mult1xN
////////
void mult1xN(
    const MemoryView<Quaternion, RAM>& a,
    const MemoryView<Quaternion, RAM>& B,
    MemoryView<Quaternion, RAM>& C);

Memory<Quaternion, RAM> mult1xN(
    const MemoryView<Quaternion, RAM>& a, 
    const MemoryView<Quaternion, RAM>& B);

void mult1xN(
    const MemoryView<Quaternion, RAM>& a,
    const MemoryView<Vector, RAM>& B, 
    MemoryView<Vector, RAM>& C);

Memory<Vector, RAM> mult1xN(
    const MemoryView<Quaternion, RAM>& a,
    const MemoryView<Vector, RAM>& B);

void mult1xN(
    const MemoryView<Transform, RAM>& t1,
    const MemoryView<Transform, RAM>& T2,
    MemoryView<Transform, RAM>& Tr);

Memory<Transform, RAM> mult1xN(
    const MemoryView<Transform, RAM>& t1,
    const MemoryView<Transform, RAM>& T2);

void mult1xN(
    const MemoryView<Transform, RAM>& t,
    const MemoryView<Vector, RAM>& X,
    MemoryView<Vector, RAM>& C);

Memory<Vector, RAM> mult1xN(
    const MemoryView<Transform, RAM>& t,
    const MemoryView<Vector, RAM>& X);

void mult1xN(
    const MemoryView<Matrix3x3, RAM>& m1,
    const MemoryView<Matrix3x3, RAM>& M2,
    MemoryView<Matrix3x3, RAM>& Mr);

Memory<Matrix3x3, RAM> mult1xN(
    const MemoryView<Matrix3x3, RAM>& m1,
    const MemoryView<Matrix3x3, RAM>& M2);

void mult1xN(
    const MemoryView<Matrix3x3, RAM>& m,
    const MemoryView<Vector, RAM>& X,
    MemoryView<Vector, RAM>& C);

Memory<Vector, RAM> mult1xN(
    const MemoryView<Matrix3x3, RAM>& m,
    const MemoryView<Vector, RAM>& X);

//////
// #add
void addNxN(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& B,
    MemoryView<Vector, RAM>& C);

Memory<Vector, RAM> addNxN(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& B);

inline Memory<Vector, RAM> operator+(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& B)
{
    return addNxN(A, B);
}

//////
// #sub
void subNxN(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& B,
    MemoryView<Vector, RAM>& C);

Memory<Vector, RAM> subNxN(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& B);

inline Memory<Vector, RAM> operator-(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& B)
{
    return subNxN(A, B);
}

void subNx1(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& b,
    MemoryView<Vector, RAM>& C);

Memory<Vector, RAM> subNx1(
    const MemoryView<Vector, RAM>& A,
    const MemoryView<Vector, RAM>& b);

void sub(
    const MemoryView<Vector, RAM>& A,
    const Vector& b,
    MemoryView<Vector, RAM>& C);

Memory<Vector, RAM> sub(
    const MemoryView<Vector, RAM>& A,
    const Vector& b);

inline Memory<Vector, RAM> operator-(
    const MemoryView<Vector, RAM>& A,
    const Vector& b)
{
    return sub(A, b);
}

/////
// #transpose
void transpose(
    const MemoryView<Matrix3x3, RAM>& A, 
    MemoryView<Matrix3x3, RAM>& B);

Memory<Matrix3x3, RAM> transpose(
    const MemoryView<Matrix3x3, RAM>& A);

void transpose(
    const MemoryView<Matrix4x4, RAM>& A,
    MemoryView<Matrix4x4, RAM>& B);

Memory<Matrix4x4, RAM> transpose(
    const MemoryView<Matrix4x4, RAM>& A);

//////
// #invert
void invert(
    const MemoryView<Matrix3x3, RAM>& A, 
    MemoryView<Matrix3x3, RAM>& B);

Memory<Matrix3x3, RAM> invert(
    const MemoryView<Matrix3x3, RAM>& A);

void invert(
    const MemoryView<Matrix4x4, RAM>& A,
    MemoryView<Matrix4x4, RAM>& B);

Memory<Matrix4x4, RAM> invert(
    const MemoryView<Matrix4x4, RAM>& A);

void invert(
    const MemoryView<Transform, RAM>& A,
    MemoryView<Transform, RAM>& B);

Memory<Transform, RAM> invert(
    const MemoryView<Transform, RAM>& A);

////////
// #pack
void pack(
    const MemoryView<Matrix3x3, RAM>& R,
    const MemoryView<Vector, RAM>& t,
    MemoryView<Transform, RAM>& T);

void pack(
    const MemoryView<Quaternion, RAM>& R,
    const MemoryView<Vector, RAM>& t,
    MemoryView<Transform, RAM>& T);

///////
// #sum
void sum(
    const MemoryView<Vector, RAM>& X, 
    MemoryView<Vector, RAM>& res);

Memory<Vector, RAM> sum(
    const MemoryView<Vector, RAM>& X);

//////
// #mean
void mean(
    const MemoryView<Vector, RAM>& X,
    MemoryView<Vector, RAM>& res);

Memory<Vector,RAM> mean(
    const MemoryView<Vector, RAM>& X);

///////
// #cov   C = (v1 * v2.T) / N
void cov(
    const MemoryView<Vector, RAM>& v1,
    const MemoryView<Vector, RAM>& v2,
    MemoryView<Matrix3x3, RAM>& C
);

Memory<Matrix3x3, RAM> cov(
    const MemoryView<Vector, RAM>& v1,
    const MemoryView<Vector, RAM>& v2
);

/**
 * @brief decompose A = UWV* using singular value decomposition
 */
void svd(
    const MemoryView<Matrix3x3, RAM>& As,
    MemoryView<Matrix3x3, RAM>& Us,
    MemoryView<Matrix3x3, RAM>& Ws,
    MemoryView<Matrix3x3, RAM>& Vs
);

/**
 * @brief decompose A = UWV* using singular value decomposition
 * 
 * w is a vector which is the diagonal of matrix W
 */
void svd(
    const MemoryView<Matrix3x3, RAM>& As,
    MemoryView<Matrix3x3, RAM>& Us,
    MemoryView<Vector3, RAM>& ws,
    MemoryView<Matrix3x3, RAM>& Vs
);

/**
 * @brief computes the optimal transformations according to Umeyama's algorithm 
 * for a list of partitions [(m,d,C,N), ...]
 * 
 * Note: sometimes referred to as Kabsch/Umeyama
 * 
 * @param n_meas: if == 0: Resulting Transform is set to identity. Otherwise the 
 *                standard Umeyama algorithm is performed
 */
void umeyama_transform(
    MemoryView<Transform, RAM>& Ts,
    const MemoryView<Vector3, RAM>& ds,
    const MemoryView<Vector3, RAM>& ms,
    const MemoryView<Matrix3x3, RAM>& Cs,
    const MemoryView<unsigned int, RAM>& n_meas
);

Memory<Transform, RAM> umeyama_transform(
    const MemoryView<Vector3, RAM>& ds,
    const MemoryView<Vector3, RAM>& ms,
    const MemoryView<Matrix3x3, RAM>& Cs,
    const MemoryView<unsigned int, RAM>& n_meas
);

/**
 * @brief computes the optimal transformations according to Umeyama's algorithm 
 * for a list of partitions [(m,d,C,N), ...]
 * 
 * Note: sometimes referred to as Kabsch/Umeyama
 */
void umeyama_transform(
    MemoryView<Transform, RAM>& Ts,
    const MemoryView<Vector3, RAM>& ds,
    const MemoryView<Vector3, RAM>& ms,
    const MemoryView<Matrix3x3, RAM>& Cs
);

Memory<Transform, RAM> umeyama_transform(
    const MemoryView<Vector3, RAM>& ds,
    const MemoryView<Vector3, RAM>& ms,
    const MemoryView<Matrix3x3, RAM>& Cs
);

// def mean_rotation_quat(Rs, weights=None):
//     """
//     Rs: list/array of 3x3 rotation matrices
//     weights: optional nonnegative weights
//     """
//     n = len(Rs)
//     if weights is None:
//         weights = np.ones(n)
//     weights = np.asarray(weights) / np.sum(weights)
//     # Convert to quaternions (w,x,y,z)
//     Q = np.stack([quat_from_R(R) for R in Rs], axis=0)

//     # Hemisphere correction using first quaternion as reference
//     ref = Q[0]
//     signs = np.sign(np.sum(Q * ref, axis=1))
//     signs[signs == 0] = 1.0
//     Q = Q * signs[:, None]

//     # Weighted outer-product averaging (Markley method)
//     M = np.zeros((4,4))
//     for w, q in zip(weights, Q):
//         M += w * np.outer(q, q)
//     eigvals, eigvecs = np.linalg.eigh(M)
//     q_mean = eigvecs[:, np.argmax(eigvals)]
//     if q_mean[0] < 0:  # ensure w >= 0
//         q_mean = -q_mean

Quaternion markley_mean(
  const MemoryView<Quaternion, RAM>& Qs, 
  const MemoryView<float, RAM> weights = MemoryView<float, RAM>::Empty() );

/**
 * Karcher mean on SE(3) with right-invariant metric.
 * Ts:       list of poses
 * w_opt:    optional weights (nonnegative). If null/empty => uniform.
 * tol:      stop when ||mean twist|| < tol  (twist norm in R^6)
 * max_iters:iteration cap
 */
Transform karcher_mean(const MemoryView<Transform, RAM> Ts,
                          const MemoryView<float, RAM> weights = MemoryView<float, RAM>::Empty(),
                          float tol = 1e-10f,
                          int max_iters = 50);

} // namespace rmagine

#endif // RMAGINE_MATH_MEMORY_MATH_H