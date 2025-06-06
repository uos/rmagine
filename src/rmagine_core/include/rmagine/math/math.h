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
 * @brief Math Functions for CPU Memory
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MATH_MATH_H
#define RMAGINE_MATH_MATH_H

#include <rmagine/math/types.h>
namespace rmagine
{

///////
// TODO: Integrate the following functions better / somewhere else
//////
// template<class T>
// RMAGINE_INLINE_FUNCTION
// T SQR(const T a) {return a*a;}

template<class T>
RMAGINE_INLINE_FUNCTION
T sqr(const T a) 
{
    return a*a;
};

template<class T>
RMAGINE_INLINE_FUNCTION
const T &max(const T &a, const T &b)
{
    return b > a ? (b) : (a);
}

RMAGINE_INLINE_FUNCTION
float max(const double &a, const float &b)
{
    return b > a ? (b) : float(a);
}

RMAGINE_INLINE_FUNCTION
float max(const float &a, const double &b)
{
    return b > a ? float(b) : (a);
}

template<class T>
RMAGINE_INLINE_FUNCTION
const T& min(const T &a, const T &b)
{
    return b < a ? (b) : (a);
}

RMAGINE_INLINE_FUNCTION
float min(const double &a, const float &b)
{
    return b < a ? (b) : float(a);
}

RMAGINE_INLINE_FUNCTION
float min(const float &a, const double &b)
{
    return b < a ? float(b) : (a);
}

template<class T>
RMAGINE_INLINE_FUNCTION
T sign(const T &a, const T &b)
{
    return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

RMAGINE_INLINE_FUNCTION
float sign(const float &a, const double &b)
{
    return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

RMAGINE_INLINE_FUNCTION
float sign(const double &a, const float &b)
{
  return (float)(b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a));
}

template<class T>
RMAGINE_INLINE_FUNCTION
void swap(T &a, T &b)
{
  T dum=a; a=b; b=dum;
}

template<typename T>
RMAGINE_INLINE_FUNCTION
T pythag(const T a, const T b)
{
  const T absa = abs(a);
  const T absb = abs(b);
  return (absa > absb ? absa * sqrt(1.0+sqr(absb/absa)) :
      (absb == 0.0 ? 0.0 : absb * sqrt(1.0+sqr(absa/absb))));
}

template<typename DataT>
Vector3_<DataT> min(const Vector3_<DataT>& a, const Vector3_<DataT>& b);

template<typename DataT>
Vector3_<DataT> max(const Vector3_<DataT>& a, const Vector3_<DataT>& b);

} // namespace rmagine

#include "math.tcc"

#endif // RMAGINE_MATH_MATH_H