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
 * @brief SVD solver for CPU Memory
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MATH_SVD2_HPP
#define RMAGINE_MATH_SVD2_HPP

#include <rmagine/types/Memory.hpp>
#include <rmagine/math/types.h>
#include <memory>

namespace rmagine {


template<class T>
inline T SQR(const T a) {return a*a;}

template<class T>
inline const T &MAX(const T &a, const T &b)
    {return b > a ? (b) : (a);}

inline float MAX(const double &a, const float &b)
    {return b > a ? (b) : float(a);}

inline float MAX(const float &a, const double &b)
    {return b > a ? float(b) : (a);}

template<class T>
inline const T &MIN(const T &a, const T &b)
    {return b < a ? (b) : (a);}

inline float MIN(const double &a, const float &b)
    {return b < a ? (b) : float(a);}

inline float MIN(const float &a, const double &b)
    {return b < a ? float(b) : (a);}

template<class T>
inline T SIGN(const T &a, const T &b)
    {return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}

inline float SIGN(const float &a, const double &b)
    {return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);}

inline float SIGN(const double &a, const float &b)
    {return (float)(b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a));}

template<class T>
inline void SWAP(T &a, T &b)
    {T dum=a; a=b; b=dum;}

template<typename T>
inline T PYTHAG(const T a, const T b)
{
    T absa = abs(a);
    T absb = abs(b);
    return (absa > absb ? absa * sqrt(1.0+SQR(absb/absa)) :
        (absb == 0.0 ? 0.0 : absb * sqrt(1.0+SQR(absa/absb))));
}


class SVD2
{
public:
    SVD2();
    ~SVD2();

    void decompose(const Matrix3x3& a);

    Matrix3x3 u,v;
	Vector3 w;
private:
    
    void reorder();
    float pythag(const float a, const float b);

    // int m, n;
	float eps, tsh;
};

// Numerical Recipes
// M = MatrixT::rows()
// N = MatrixT::cols()
template<typename MatrixT>
struct svd_dims {
    using U = MatrixT; // same as input
    using w = Matrix_<typename MatrixT::Type, MatrixT::cols(), 1>;
    using W = Matrix_<typename MatrixT::Type, MatrixT::cols(), MatrixT::cols()>;
    using V = Matrix_<typename MatrixT::Type, MatrixT::cols(), MatrixT::cols()>;
};

// Wikipedia: M = USV*
// - M: mxn
// - U: mxm
// - S: mxn
// - V*: nxn - V: nxn

/**
 *
 * @brief own SVD implementation 
 *
 */
template<typename DataT, unsigned int Rows, unsigned int Cols>
void svd(
    const Matrix_<DataT, Rows, Cols>& A,
    Matrix_<DataT, Rows, Cols>& U,
    Matrix_<DataT, Cols, 1>& w, // vector version (Cols should be something with max)
    Matrix_<DataT, Cols, Cols>& V
);

template<typename DataT, unsigned int Rows, unsigned int Cols>
void svd(
    const Matrix_<DataT, Rows, Cols>& A, 
    Matrix_<DataT, Rows, Cols>& U,
    Matrix_<DataT, Cols, Cols>& W, // matrix
    Matrix_<DataT, Cols, Cols>& V
);

using SVD2Ptr = std::shared_ptr<SVD2>;

} // namespace rmagine

#include "SVD2.tcc"

#endif // RMAGINE_MATH_SVD2_HPP
