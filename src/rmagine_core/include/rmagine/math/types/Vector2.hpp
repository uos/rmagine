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
 * @brief Vector2
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_VECTOR2_HPP
#define RMAGINE_MATH_VECTOR2_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>
#include <initializer_list>

namespace rmagine
{

template<typename DataT>
struct Vector2_
{
    DataT x;
    DataT y;

    RMAGINE_FUNCTION
    static Vector2_<DataT> NaN()
    {
        return {NAN, NAN};
    }

    RMAGINE_FUNCTION
    static Vector2_<DataT> Zeros()
    {
        return {static_cast<DataT>(0), static_cast<DataT>(0)};
    }

    RMAGINE_FUNCTION
    static Vector2_<DataT> Ones()
    {
        return {static_cast<DataT>(1), static_cast<DataT>(1)};
    }

    // FUNCTIONS
    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> add(const Vector2_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    void addInplace(const Vector2_<DataT>& b);

    RMAGINE_INLINE_FUNCTION
    void addInplace(volatile Vector2_<DataT>& b) volatile;

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> sub(const Vector2_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    void subInplace(const Vector2_<DataT>& b);

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> negate() const;

    RMAGINE_INLINE_FUNCTION
    void negateInplace();

    RMAGINE_INLINE_FUNCTION
    DataT dot(const Vector2_<DataT>& b) const;

    /**
     * @brief product
     */
    RMAGINE_INLINE_FUNCTION
    DataT mult(const Vector2_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> mult(const DataT& s) const;    

    RMAGINE_INLINE_FUNCTION
    void multInplace(const DataT& s);

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, 2, 2> multT(const Vector2_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> div(const DataT& s) const;

    RMAGINE_INLINE_FUNCTION
    void divInplace(const DataT& s);

    /**
     * @brief vector from this to another
     * 
     * return o - this
    */
    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> to(const Vector2_<DataT>& o) const;

    RMAGINE_INLINE_FUNCTION
    DataT l2normSquared() const;

    RMAGINE_INLINE_FUNCTION
    DataT l2norm() const;

    RMAGINE_INLINE_FUNCTION
    DataT sum() const;

    RMAGINE_INLINE_FUNCTION
    DataT prod() const;

    RMAGINE_INLINE_FUNCTION
    DataT l1norm() const;

    RMAGINE_INLINE_FUNCTION
    void setZeros();

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> operator+(const Vector2_<DataT>& b) const
    {
        return add(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT>& operator+=(const Vector2_<DataT>& b)
    {
        addInplace(b);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> operator-(const Vector2_<DataT>& b) const
    {
        return sub(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT>& operator-=(const Vector2_<DataT>& b)
    {
        subInplace(b);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> operator-() const
    {
        return negate();
    }

    RMAGINE_INLINE_FUNCTION
    DataT operator*(const Vector2_<DataT>& b) const
    {
        return mult(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> operator*(const DataT& s) const 
    {
        return mult(s);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT>& operator*=(const DataT& s)
    {
        multInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> operator/(const DataT& s) const 
    {
        return div(s);
    }

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT>& operator/=(const DataT& s)
    {
        divInplace(s);
        return *this;
    }

    template<typename ConvT>
    RMAGINE_INLINE_FUNCTION
    Vector2_<ConvT> cast() const;
};


} // namespace rmagine

#include "Vector2.tcc"

#endif // RMAGINE_MATH_VECTOR2_HPP