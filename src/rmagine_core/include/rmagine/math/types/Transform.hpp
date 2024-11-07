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
 * @brief Transform
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_TRANSFORM_HPP
#define RMAGINE_MATH_TRANSFORM_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>
#include "Vector3.hpp"
#include "Quaternion.hpp"

namespace rmagine
{

/**
 * @brief Transform type
 * 
 * Consists of rotational part represented as @link rmagine::Quaternion Quaternion @endlink 
 * and a translational part represented as @link rmagine::Vector3 Vector3 @endlink  
 * 
 * Additionally it contains a timestamp uint32_t
 * 
 */
template<typename DataT>
struct Transform_
{
    // DATA
    Quaternion_<DataT> R;
    Vector3_<DataT> t;
    uint32_t stamp;

    // FUNCTIONS
    RMAGINE_FUNCTION
    static Transform_<DataT> Identity()
    {
        Transform_<DataT> ret;
        ret.setIdentity();
        return ret;
    }

    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    /**
     * @brief Setting the transform from an 4x4 transformation matrix
     * WARNING matrix must be isometric, i.e. must only contain
     * rotational and translational parts (not scale). Otherwise,
     * use "decompose" function
     */
    RMAGINE_INLINE_FUNCTION
    void set(const Matrix_<DataT, 4, 4>& M);

    RMAGINE_INLINE_FUNCTION
    Transform_<DataT> inv() const;

    /**
     * @brief Transform of type T3 = this*T2
     * 
     * @param T2 Other transform
     */
    RMAGINE_INLINE_FUNCTION
    Transform_<DataT> mult(const Transform_<DataT>& T2) const;

    /**
     * @brief Transform of type this = this * T2
     * 
     * @param T2 Other transform
     */
    RMAGINE_INLINE_FUNCTION
    void multInplace(const Transform_<DataT>& T2);

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> mult(const Vector3_<DataT>& v) const;

    /**
     * @brief computes the diff transform Td to another transform T2
     * so that: this * Td = T2
     * 
     * or difference between other and this rotation
     * Td = T2 - this
    */
    RMAGINE_INLINE_FUNCTION
    Transform_<DataT> to(const Transform_<DataT>& T2) const;

    RMAGINE_INLINE_FUNCTION
    Transform_<DataT> pow(const DataT& exp) const;

    RMAGINE_INLINE_FUNCTION
    Transform_<DataT> operator~() const
    {
        return inv();
    }

    RMAGINE_INLINE_FUNCTION
    Transform_<DataT> operator*(const Transform_<DataT>& T2) const 
    {
        return mult(T2);
    }

    RMAGINE_INLINE_FUNCTION
    Transform_<DataT>& operator*=(const Transform_<DataT>& T2)
    {
        multInplace(T2);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator*(const Vector3_<DataT>& v) const
    {
        return mult(v);
    }

    /////////////////////
    // CASTING

    /**
     * @brief Transform -> Matrix4x4
     * 
     * @return Matrix_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    operator Matrix_<DataT, 4, 4>() const;

    /**
     * @brief Internal data type cast
     */
    template<typename ConvT>
    Transform_<ConvT> cast() const
    {
        return {
            R.template cast<ConvT>(),
            t.template cast<ConvT>(),
            stamp
        };
    }
};

} // namespace rmagine

#include "Transform.tcc"

#endif // RMAGINE_MATH_TRANSFORM_HPP