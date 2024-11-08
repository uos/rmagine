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
 * @brief Vector3
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */
#ifndef RMAGINE_MATH_VECTOR3_HPP
#define RMAGINE_MATH_VECTOR3_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>
#include <initializer_list>
#include <limits>

namespace rmagine
{

template<typename DataT>
struct Vector3_
{
    DataT x;
    DataT y;
    DataT z;

    RMAGINE_FUNCTION
    static Vector3_<DataT> NaN()
    {
        return {NAN, NAN, NAN};
    }

    RMAGINE_FUNCTION
    static Vector3_<DataT> Zeros()
    {
        return {static_cast<DataT>(0), static_cast<DataT>(0), static_cast<DataT>(0)};
    }

    RMAGINE_FUNCTION
    static Vector3_<DataT> Ones()
    {
        return {static_cast<DataT>(1), static_cast<DataT>(1), static_cast<DataT>(1)};
    }

    RMAGINE_FUNCTION
    static Vector3_<DataT> Max()
    {
        return {std::numeric_limits<DataT>::max(), std::numeric_limits<DataT>::max(), std::numeric_limits<DataT>::max()};
    }

    RMAGINE_FUNCTION
    static Vector3_<DataT> Min()
    {
        return {std::numeric_limits<DataT>::lowest(), std::numeric_limits<DataT>::lowest(), std::numeric_limits<DataT>::lowest()};
    }

    // FUNCTIONS
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> add(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    volatile Vector3_<DataT> add(volatile Vector3_<DataT>& b) const volatile;
    
    RMAGINE_INLINE_FUNCTION
    void addInplace(const Vector3_<DataT>& b);

    RMAGINE_INLINE_FUNCTION
    void addInplace(volatile Vector3_<DataT>& b) volatile;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> sub(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> negate() const;

    RMAGINE_INLINE_FUNCTION
    void negateInplace();

    RMAGINE_INLINE_FUNCTION
    void subInplace(const Vector3_<DataT>& b);

    RMAGINE_INLINE_FUNCTION
    DataT dot(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> cross(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    DataT mult(const Vector3_<DataT>& b) const;
    
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> multEwise(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> mult(const DataT& s) const;

    RMAGINE_INLINE_FUNCTION
    volatile Vector3_<DataT> mult(const DataT& s) const volatile;

    RMAGINE_INLINE_FUNCTION
    void multInplace(const DataT& s);

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, 3, 3> multT(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> div(const DataT& s) const;

    RMAGINE_INLINE_FUNCTION
    void divInplace(const DataT& s);

    /**
     * @brief vector from this to another
     * 
     * return o - this
    */
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> to(const Vector3_<DataT>& o) const;

    RMAGINE_INLINE_FUNCTION
    DataT l2normSquared() const;

    /**
     * @brief sqrt(x*x + y*y + z*z)
     * 
     * @return norm
     */
    RMAGINE_INLINE_FUNCTION
    DataT l2norm() const; 

    RMAGINE_INLINE_FUNCTION
    DataT sum() const;

    RMAGINE_INLINE_FUNCTION
    DataT prod() const;
    
    RMAGINE_INLINE_FUNCTION
    DataT l1norm() const;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> normalize() const;

    RMAGINE_INLINE_FUNCTION
    void normalizeInplace();

    RMAGINE_INLINE_FUNCTION
    void setZeros();

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator+(const Vector3_<DataT>& b) const
    {
        return add(b);
    }

    RMAGINE_INLINE_FUNCTION
    volatile Vector3_<DataT> operator+(volatile Vector3_<DataT> b) const volatile
    {
        return add(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT>& operator+=(const Vector3_<DataT>& b)
    {
        addInplace(b);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    volatile Vector3_<DataT>& operator+=(volatile Vector3_<DataT>& b) volatile
    {
        addInplace(b);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator-(const Vector3_<DataT>& b) const
    {
        return sub(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT>& operator-=(const Vector3_<DataT>& b)
    {
        subInplace(b);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator-() const
    {
        return negate();
    }

    RMAGINE_INLINE_FUNCTION
    DataT operator*(const Vector3_<DataT>& b) const
    {
        return mult(b);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator*(const DataT& s) const 
    {
        return mult(s);
    }

    RMAGINE_INLINE_FUNCTION
    volatile Vector3_<DataT> operator*(const DataT& s) const volatile
    {
        return mult(s);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT>& operator*=(const DataT& s)
    {
        multInplace(s);
        return *this;
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator/(const DataT& s) const 
    {
        return div(s);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator/=(const DataT& s)
    {
        divInplace(s);
        return *this;
    }

    // RMAGINE_INLINE_FUNCTION
    // volatile Vector3_<DataT>& operator=(volatile Vector3_<DataT> b) volatile
    // {
    //     x = b.x;
    //     y = b.y;
    //     z = b.z;
    //     return *this;
    // }

    ////////////////////////
    // CASTING
    template<typename ConvT>
    RMAGINE_INLINE_FUNCTION
    Vector3_<ConvT> cast() const;

    ///////////////////////
    // DEPRECATED FUNCTIONS
    [[deprecated("Use multEwise() instead.")]]
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> mult_ewise(const Vector3_<DataT>& b) const
    {   
        return multEwise(b);
    }

    // Use with care!
    RMAGINE_INLINE_FUNCTION
    DataT& operator[](const size_t& idx)
    {
        return *(&x + idx);
    }

    RMAGINE_INLINE_FUNCTION
    const DataT& operator[](const size_t& idx) const
    {
        return *(&x + idx);
    }

    RMAGINE_INLINE_FUNCTION
    DataT& operator()(const size_t& idx)
    {
        return *(&x + idx);
    }

    RMAGINE_INLINE_FUNCTION
    const DataT& operator()(const size_t& idx) const
    {
        return *(&x + idx);
    }
};

} // namespace rmagine

#include "Vector3.tcc"

#endif // RMAGINE_MATH_VECTOR3_HPP
