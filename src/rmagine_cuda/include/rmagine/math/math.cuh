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
 * @brief Math Function for CUDA Memory
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MATH_MATH_CUH
#define RMAGINE_MATH_MATH_CUH

#include <rmagine/math/types.h>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/math/math_batched.cuh>

namespace rmagine 
{

/////////////
// #multNxN
////////
void multNxN(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Quaternion, VRAM_CUDA>& B,
    MemoryView<Quaternion, VRAM_CUDA>& C);

Memory<Quaternion, VRAM_CUDA> multNxN(
    const MemoryView<Quaternion, VRAM_CUDA>& A, 
    const MemoryView<Quaternion, VRAM_CUDA>& B);

inline Memory<Quaternion, VRAM_CUDA> operator*(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Quaternion, VRAM_CUDA>& B)
{
    return multNxN(A, B);
}

void multNxN(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b, 
    MemoryView<Vector, VRAM_CUDA>& c);

Memory<Vector, VRAM_CUDA> multNxN(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b);

void multNxN(
    const MemoryView<Transform, VRAM_CUDA>& T1,
    const MemoryView<Transform, VRAM_CUDA>& T2,
    MemoryView<Transform, VRAM_CUDA>& Tr);

Memory<Transform, VRAM_CUDA> multNxN(
    const MemoryView<Transform, VRAM_CUDA>& T1,
    const MemoryView<Transform, VRAM_CUDA>& T2);

void multNxN(
    const MemoryView<Transform, VRAM_CUDA>& T,
    const MemoryView<Vector, VRAM_CUDA>& x,
    MemoryView<Vector, VRAM_CUDA>& c);

Memory<Vector, VRAM_CUDA> multNxN(
    const MemoryView<Transform, VRAM_CUDA>& T,
    const MemoryView<Vector, VRAM_CUDA>& x);

void multNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& M2,
    MemoryView<Matrix3x3, VRAM_CUDA>& Mr);

Memory<Matrix3x3, VRAM_CUDA> multNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& M2);

void multNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& M2,
    MemoryView<Quaternion, VRAM_CUDA>& Qres);

void multNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x,
    MemoryView<Vector, VRAM_CUDA>& c);

Memory<Vector, VRAM_CUDA> multNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x);


/////////////
// #multNx1
////////
void multNx1(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Quaternion, VRAM_CUDA>& b,
    MemoryView<Quaternion, VRAM_CUDA>& C);

Memory<Quaternion, VRAM_CUDA> multNx1(
    const MemoryView<Quaternion, VRAM_CUDA>& A, 
    const MemoryView<Quaternion, VRAM_CUDA>& b);

void multNx1(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b, 
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> multNx1(
    const MemoryView<Quaternion, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b);

void multNx1(
    const MemoryView<Transform, VRAM_CUDA>& T1,
    const MemoryView<Transform, VRAM_CUDA>& t2,
    MemoryView<Transform, VRAM_CUDA>& Tr);

Memory<Transform, VRAM_CUDA> multNx1(
    const MemoryView<Transform, VRAM_CUDA>& T1,
    const MemoryView<Transform, VRAM_CUDA>& t2);

void multNx1(
    const MemoryView<Transform, VRAM_CUDA>& T,
    const MemoryView<Vector, VRAM_CUDA>& x,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> multNx1(
    const MemoryView<Transform, VRAM_CUDA>& T,
    const MemoryView<Vector, VRAM_CUDA>& x);

void multNx1(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& m2,
    MemoryView<Matrix3x3, VRAM_CUDA>& Mr);

Memory<Matrix3x3, VRAM_CUDA> multNx1(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& m2);

void multNx1(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> multNx1(
    const MemoryView<Matrix3x3, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x);

void multNx1(
    const MemoryView<Matrix4x4, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> multNx1(
    const MemoryView<Matrix4x4, VRAM_CUDA>& M,
    const MemoryView<Vector, VRAM_CUDA>& x);

/////////////
// #mult1xN
////////
void mult1xN(
    const MemoryView<Quaternion, VRAM_CUDA>& a,
    const MemoryView<Quaternion, VRAM_CUDA>& B,
    MemoryView<Quaternion, VRAM_CUDA>& C);

Memory<Quaternion, VRAM_CUDA> mult1xN(
    const MemoryView<Quaternion, VRAM_CUDA>& a, 
    const MemoryView<Quaternion, VRAM_CUDA>& B);

void mult1xN(
    const MemoryView<Quaternion, VRAM_CUDA>& a,
    const MemoryView<Vector, VRAM_CUDA>& B, 
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> mult1xN(
    const MemoryView<Quaternion, VRAM_CUDA>& a,
    const MemoryView<Vector, VRAM_CUDA>& B);

void mult1xN(
    const MemoryView<Transform, VRAM_CUDA>& t1,
    const MemoryView<Transform, VRAM_CUDA>& T2,
    MemoryView<Transform, VRAM_CUDA>& Tr);

Memory<Transform, VRAM_CUDA> mult1xN(
    const MemoryView<Transform, VRAM_CUDA>& t1,
    const MemoryView<Transform, VRAM_CUDA>& T2);

void mult1xN(
    const MemoryView<Transform, VRAM_CUDA>& t,
    const MemoryView<Vector, VRAM_CUDA>& X,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> mult1xN(
    const MemoryView<Transform, VRAM_CUDA>& t,
    const MemoryView<Vector, VRAM_CUDA>& X);

void mult1xN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& m1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& M2,
    MemoryView<Matrix3x3, VRAM_CUDA>& Mr);

Memory<Matrix3x3, VRAM_CUDA> mult1xN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& m1,
    const MemoryView<Matrix3x3, VRAM_CUDA>& M2);

void mult1xN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& m,
    const MemoryView<Vector, VRAM_CUDA>& X,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> mult1xN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& m,
    const MemoryView<Vector, VRAM_CUDA>& X);

void mult1xN(
    const MemoryView<Matrix4x4, VRAM_CUDA>& m,
    const MemoryView<Vector, VRAM_CUDA>& X,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> mult1xN(
    const MemoryView<Matrix4x4, VRAM_CUDA>& m,
    const MemoryView<Vector, VRAM_CUDA>& X);

//////
// #add
void addNxN(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& B,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> addNxN(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& B);

inline Memory<Vector, VRAM_CUDA> operator+(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& B)
{
    return addNxN(A, B);
}



void addNxN(
    const MemoryView<float, VRAM_CUDA>& a,
    const MemoryView<float, VRAM_CUDA>& b,
    MemoryView<float, VRAM_CUDA>& c);

Memory<float, VRAM_CUDA> addNxN(
    const MemoryView<float, VRAM_CUDA>& a,
    const MemoryView<float, VRAM_CUDA>& b);

inline Memory<float, VRAM_CUDA> operator+(
    const MemoryView<float, VRAM_CUDA>& A,
    const MemoryView<float, VRAM_CUDA>& B)
{
    return addNxN(A, B);
}




//////
// #sub
void subNxN(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& B,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> subNxN(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& B);

inline Memory<Vector, VRAM_CUDA> operator-(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& B)
{
    return subNxN(A, B);
}

void subNx1(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> subNx1(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<Vector, VRAM_CUDA>& b);


/////
// #transpose
void transpose(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    MemoryView<Matrix3x3, VRAM_CUDA>& B);

Memory<Matrix3x3, VRAM_CUDA> transpose(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A);

void transpose(
    const MemoryView<Matrix4x4, VRAM_CUDA>& A,
    MemoryView<Matrix4x4, VRAM_CUDA>& B);

Memory<Matrix4x4, VRAM_CUDA> transpose(
    const MemoryView<Matrix4x4, VRAM_CUDA>& A);

///////
// #transposeInplace
void transposeInplace(
    MemoryView<Matrix3x3, VRAM_CUDA>& A);

//////
// #invert
void invert(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    MemoryView<Matrix3x3, VRAM_CUDA>& B);

Memory<Matrix3x3, VRAM_CUDA> invert(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A);

void invert(
    const MemoryView<Matrix4x4, VRAM_CUDA>& A,
    MemoryView<Matrix4x4, VRAM_CUDA>& B);

Memory<Matrix4x4, VRAM_CUDA> invert(
    const MemoryView<Matrix4x4, VRAM_CUDA>& A);

void invert(
    const MemoryView<Transform, VRAM_CUDA>& A,
    MemoryView<Transform, VRAM_CUDA>& B);

Memory<Transform, VRAM_CUDA> invert(
    const MemoryView<Transform, VRAM_CUDA>& A);

//////
// #divNxN
void divNxN(
    const MemoryView<Vector, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> divNxN(
    const MemoryView<Vector, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B);

inline Memory<Vector, VRAM_CUDA> operator/(
    const MemoryView<Vector, VRAM_CUDA>& A,
    const MemoryView<unsigned int, VRAM_CUDA>& B)
{
    return divNxN(A, B);
}

void divNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B, 
    MemoryView<Matrix3x3, VRAM_CUDA>& C);

Memory<Matrix3x3, VRAM_CUDA> divNxN(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B);

inline Memory<Matrix3x3, VRAM_CUDA> operator/(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B)
{
    return divNxN(A, B);
}

///////
// #divNxNIgnoreZeros
void divNxNIgnoreZeros(
    const MemoryView<Vector, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B,
    MemoryView<Vector, VRAM_CUDA>& C);

Memory<Vector, VRAM_CUDA> divNxNIgnoreZeros(
    const MemoryView<Vector, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B);

void divNxNIgnoreZeros(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B,
    MemoryView<Matrix3x3, VRAM_CUDA>& C);

Memory<Matrix3x3, VRAM_CUDA> divNxNIgnoreZeros(
    const MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B);

void divNxNIgnoreZeros(
    const MemoryView<float, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B,
    MemoryView<float, VRAM_CUDA>& C);

Memory<float, VRAM_CUDA> divNxNIgnoreZeros(
    const MemoryView<float, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B);

////////
// #divNxNInplace
void divNxNInplace(
    MemoryView<Vector, VRAM_CUDA>& A, 
    const MemoryView<float, VRAM_CUDA>& B);

void divNxNInplace(
    MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const MemoryView<unsigned int, VRAM_CUDA>& B);

////////
// #divNx1Inplace
void divNx1Inplace(
    MemoryView<Matrix3x3, VRAM_CUDA>& A, 
    const unsigned int& B);

void divNx1Inplace(
    MemoryView<Vector, VRAM_CUDA>& A, 
    const unsigned int& B);

////////
// #convert
void convert(const MemoryView<uint8_t, VRAM_CUDA>& from, 
    MemoryView<float, VRAM_CUDA>& to);

void convert(const MemoryView<bool, VRAM_CUDA>& from, 
    MemoryView<unsigned int, VRAM_CUDA>& to);

void convert(const MemoryView<unsigned int, VRAM_CUDA>& from, 
    MemoryView<bool, VRAM_CUDA>& to);

////////
// #pack
void pack(
    const MemoryView<Matrix3x3, VRAM_CUDA>& R,
    const MemoryView<Vector, VRAM_CUDA>& t,
    MemoryView<Transform, VRAM_CUDA>& T);

void pack(
    const MemoryView<Quaternion, VRAM_CUDA>& R,
    const MemoryView<Vector, VRAM_CUDA>& t,
    MemoryView<Transform, VRAM_CUDA>& T);


////////
// #multNxNTransposed
void multNxNTransposed(
    const MemoryView<Vector, VRAM_CUDA>& m1,
    const MemoryView<Vector, VRAM_CUDA>& m2,
    MemoryView<Matrix3x3, VRAM_CUDA>& Cs);

Memory<Matrix3x3, VRAM_CUDA> multNxNTransposed(
    const MemoryView<Vector, VRAM_CUDA>& m1,
    const MemoryView<Vector, VRAM_CUDA>& m2);

void multNxNTransposed(
    const MemoryView<Vector, VRAM_CUDA>& m1,
    const MemoryView<Vector, VRAM_CUDA>& m2,
    const MemoryView<bool, VRAM_CUDA>& mask,
    MemoryView<Matrix3x3, VRAM_CUDA>& Cs);
    
Memory<Matrix3x3, VRAM_CUDA> multNxNTransposed(
    const MemoryView<Vector, VRAM_CUDA>& m1,
    const MemoryView<Vector, VRAM_CUDA>& m2,
    const MemoryView<bool, VRAM_CUDA>& mask);


///////
// #normalize
void normalizeInplace(MemoryView<Quaternion, VRAM_CUDA>& q);

///////
// #setter
void setIdentity(MemoryView<Quaternion, VRAM_CUDA>& qs);

void setIdentity(MemoryView<Transform, VRAM_CUDA>& Ts);

void setIdentity(MemoryView<Matrix3x3, VRAM_CUDA>& Ms);

void setIdentity(MemoryView<Matrix4x4, VRAM_CUDA>& Ms);

void setZeros(MemoryView<Matrix3x3, VRAM_CUDA>& Ms);

void setZeros(MemoryView<Matrix4x4, VRAM_CUDA>& Ms);


//////////
// #sum
void sum(
    const MemoryView<Vector, VRAM_CUDA>& data,
    MemoryView<Vector, VRAM_CUDA>& s);

Memory<Vector, VRAM_CUDA> sum(
    const MemoryView<Vector, VRAM_CUDA>& data);

void sum(
    const MemoryView<int, VRAM_CUDA>& data,
    MemoryView<int, VRAM_CUDA>& s);

Memory<int, VRAM_CUDA> sum(
    const MemoryView<int, VRAM_CUDA>& data);

//////////
// #mean
void mean(
    const MemoryView<Vector, VRAM_CUDA>& X,
    MemoryView<Vector, VRAM_CUDA>& res);

Memory<Vector, VRAM_CUDA> mean(
    const MemoryView<Vector, VRAM_CUDA>& X);

//////////
// v1: from
// v2: to
// #cov   C = (v1 * v2.T) / N
void cov(
    const MemoryView<Vector, VRAM_CUDA>& v1,
    const MemoryView<Vector, VRAM_CUDA>& v2,
    MemoryView<Matrix3x3, VRAM_CUDA>& C
);

Memory<Matrix3x3, VRAM_CUDA> cov(
    const MemoryView<Vector, VRAM_CUDA>& v1,
    const MemoryView<Vector, VRAM_CUDA>& v2
);

/**
 * @brief decompose A = UWV* using singular value decomposition
 */
void svd(
    const MemoryView<Matrix3x3, VRAM_CUDA>& As,
    MemoryView<Matrix3x3, VRAM_CUDA>& Us,
    MemoryView<Matrix3x3, VRAM_CUDA>& Ws,
    MemoryView<Matrix3x3, VRAM_CUDA>& Vs
);

/**
 * @brief decompose A = UWV* using singular value decomposition
 * 
 * w is a vector which is the diagonal of matrix W
 */
void svd(
    const MemoryView<Matrix3x3, VRAM_CUDA>& As,
    MemoryView<Matrix3x3, VRAM_CUDA>& Us,
    MemoryView<Vector3, VRAM_CUDA>& ws,
    MemoryView<Matrix3x3, VRAM_CUDA>& Vs
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
    MemoryView<Transform, VRAM_CUDA>& Ts,
    const MemoryView<Vector3, VRAM_CUDA>& ds,
    const MemoryView<Vector3, VRAM_CUDA>& ms,
    const MemoryView<Matrix3x3, VRAM_CUDA>& Cs,
    const MemoryView<unsigned int, VRAM_CUDA>& n_meas
);

Memory<Transform, VRAM_CUDA> umeyama_transform(
    const MemoryView<Vector3, VRAM_CUDA>& ds,
    const MemoryView<Vector3, VRAM_CUDA>& ms,
    const MemoryView<Matrix3x3, VRAM_CUDA>& Cs,
    const MemoryView<unsigned int, VRAM_CUDA>& n_meas
);

/**
 * @brief computes the optimal transformations according to Umeyama's algorithm 
 * for a list of partitions [(m,d,C,N), ...]
 * 
 * Note: sometimes referred to as Kabsch/Umeyama
 */
void umeyama_transform(
    MemoryView<Transform, VRAM_CUDA>& Ts,
    const MemoryView<Vector3, VRAM_CUDA>& ds,
    const MemoryView<Vector3, VRAM_CUDA>& ms,
    const MemoryView<Matrix3x3, VRAM_CUDA>& Cs
);

Memory<Transform, VRAM_CUDA> umeyama_transform(
    const MemoryView<Vector3, VRAM_CUDA>& ds,
    const MemoryView<Vector3, VRAM_CUDA>& ms,
    const MemoryView<Matrix3x3, VRAM_CUDA>& Cs
);


} // namespace rmagine

#endif // RMAGINE_MATH_MATH_CUH