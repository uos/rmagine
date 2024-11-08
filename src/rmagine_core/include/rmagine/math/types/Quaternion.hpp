/*
 * Copyright (c) 2024, University Osnabrück
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
 * @brief Quaternion
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_QUATERNION_HPP
#define RMAGINE_MATH_QUATERNION_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>
#include <initializer_list>

namespace rmagine
{


/**
 * @brief Quaternion_<DataT> type
 * 
 */
template<typename DataT>
struct Quaternion_
{
    // Data
    DataT x;
    DataT y;
    DataT z;
    DataT w;

    RMAGINE_FUNCTION
    static Quaternion_<DataT> Identity()
    {
        Quaternion_<DataT> ret;
        ret.setIdentity();
        return ret;
    }

    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    /**
     * @brief Invert this Quaternion
     * 
     * @return Quaternion_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> inv() const;

    RMAGINE_INLINE_FUNCTION
    void invInplace();

    /**
     * @brief Multiply quaternion
     * 
     * @param q2 
     * @return Quaternion_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> mult(const Quaternion_<DataT>& q2) const;

    RMAGINE_INLINE_FUNCTION
    void multInplace(const Quaternion_<DataT>& q2);

    /**
     * @brief computes the diff quaternion qd to another quaternion q2
     * so that: this * qd = q2  
     * or q10 * q21 = q20
     * 
     * or difference between other and this rotation
     * res = q2 - this
    */
    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> to(const Quaternion_<DataT>& q2) const;

    /**
     * @brief Rotate a vector with this quaternion
     * 
     * e.g. Hamiltion product
     * 
     * @param p 
     * @return Vector 
     */
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> mult(const Vector3_<DataT>& p) const;

    /**
     * 
     * Element-wise product
    */
    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> mult(const DataT& scalar) const;

    RMAGINE_INLINE_FUNCTION
    DataT dot(const Quaternion_<DataT>& q) const;

    /**
     * @brief apply rotations multiple times
     * 
     * simplest case:
     * R * R * x = R^2 * x
     * 
     * Or half a rotation:
     * R^(0.5) * x
     * 
     * Or parts of different rotations (Slerp)
     * Ra^(a) * Rb^(b)
    */
    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> pow(const DataT& exp) const;

    RMAGINE_INLINE_FUNCTION
    DataT l2normSquared() const;

    RMAGINE_INLINE_FUNCTION
    DataT l2norm() const;

    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> normalize() const;
    
    RMAGINE_INLINE_FUNCTION
    void normalizeInplace();

    RMAGINE_INLINE_FUNCTION
    void set(const Matrix_<DataT, 3, 3>& M);

    RMAGINE_INLINE_FUNCTION
    void set(const EulerAngles_<DataT>& e);

    // TODO: Quaternion from rotation around an axis v by an angle a
    // RMAGINE_INLINE_FUNCTION
    // void set(const Vector3& v, float a);

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> operator~() const 
    {
        return inv();
    }

    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> operator*(const Quaternion_<DataT>& q2) const 
    {
        return mult(q2);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator*(const Vector3_<DataT>& p) const
    {
        return mult(p);
    }

    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT>& operator*=(const Quaternion_<DataT>& q2)
    {
        multInplace(q2);
        return *this;
    }

    /////////////////////
    // CASTING

    /**
     * @brief Quaternion -> EulerAngles
     * 
     * @return EulerAngles_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    operator EulerAngles_<DataT>() const;

    /**
     * @brief Quaternion -> Rotation Matrix
     * 
     * @return Matrix_<DataT, 3, 3> 
     */
    RMAGINE_INLINE_FUNCTION
    operator Matrix_<DataT, 3, 3>() const;

    /**
     * @brief Data Type cast to ConvT
     * 
     * @tparam ConvT 
     * @return RMAGINE_INLINE_FUNCTION 
     */
    template<typename ConvT>
    RMAGINE_INLINE_FUNCTION
    Quaternion_<ConvT> cast() const;
};

} // namespace rmagine

#include "Quaternion.tcc"

#endif // RMAGINE_MATH_QUATERNION_HPP