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
 * @brief Linear Algebra Function
 *
 * @date 19.08.2025
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2025, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_LIE_H
#define RMAGINE_MATH_LIE_H

#include "types.h"
#include "math.h"
#include <rmagine/types/shared_functions.h>
#include <rmagine/types/PointCloud.hpp>

#include <rmagine/util/prints.h>

namespace rmagine
{

/** 
 * [ω]x operator: maps a 3-vector to its skew-symmetric matrix
 */
template<typename DataT>
Matrix3x3_<DataT> omega_hat(const Vector3_<DataT>& w);

/**
 * Rodrigues' rotation formula
 * @param axis   rotation axis (does not need to be unit; will be normalized)
 * @param theta  rotation angle in radians
 * @return       3x3 rotation matrix
 */
template<typename DataT>
Matrix3x3_<DataT> rodrigues(const Vector3_<DataT>& axis, DataT theta);

/**
 * Convenience overload: Rodrigues from axis–angle vector ω = θ * k.
 * Equivalent to so3_exp(ω).
 */
template<typename DataT>
Matrix3x3_<DataT> rodrigues(const Vector3_<DataT>& omega);



/** 
 * Log map on SO(3): R -> omega (axis-angle vector) 
 */ 
template<typename DataT>
Vector3_<DataT> so3_log(
  const Matrix3x3_<DataT>& R);

template<typename DataT>
Matrix3x3_<DataT> so3_exp(
  const Vector3_<DataT> omega);

// --- Left Jacobian of SO(3) ---
// J_l(w) = I + (1-cosθ)/θ^2 [w]x + (θ - sinθ)/θ^3 [w]x^2
// Small-angle series: I - 1/2 [w]x + 1/6 [w]x^2
template<typename DataT>
Matrix3x3_<DataT> so3_left_jacobian(const Vector3_<DataT>& w);

// J_l(w)^{-1} = I - 1/2 [w]x + (1 - θ cot(θ/2))/θ^2 [w]x^2
// Small-angle series: I + 1/2 [w]x + 1/12 [w]x^2
template<typename DataT>
Matrix3x3_<DataT> so3_left_jacobian_inv(const Vector3_<DataT>& w);



// ------------------------------------------------------------------
// se3 exponential: twist (v, w) -> Transform
// ------------------------------------------------------------------
template<typename DataT>
Transform_<DataT> se3_exp(
  const Vector3_<DataT>& v, 
  const Vector3_<DataT>& w);

template<typename DataT>
std::pair<Vector3_<DataT>, Vector3_<DataT> > se3_log(
  const Transform_<DataT>& T);

} // namespace rmagine

#include "lie.tcc"

#endif // RMAGINE_MATH_LIE_H

