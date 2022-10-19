#ifndef RMAGINE_MATH_VECTOR3_HPP
#define RMAGINE_MATH_VECTOR3_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>
#include <initializer_list>

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

    // FUNCTIONS
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> add(const Vector3_<DataT>& b) const;
    
    RMAGINE_INLINE_FUNCTION
    void addInplace(const Vector3_<DataT>& b);

    RMAGINE_INLINE_FUNCTION
    void addInplace(volatile Vector3_<DataT>& b) volatile;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> sub(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> negation() const;

    RMAGINE_INLINE_FUNCTION
    void negate();

    RMAGINE_INLINE_FUNCTION
    void subInplace(const Vector3_<DataT>& b);

    RMAGINE_INLINE_FUNCTION
    DataT dot(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> cross(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    DataT mult(const Vector3_<DataT>& b) const;
    
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> mult_ewise(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> mult(const DataT& s) const;

    RMAGINE_INLINE_FUNCTION
    void multInplace(const DataT& s);

    RMAGINE_INLINE_FUNCTION
    Matrix_<DataT, 3, 3> multT(const Vector3_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> div(const DataT& s) const;

    RMAGINE_INLINE_FUNCTION
    void divInplace(const DataT& s);

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
    Vector3_<DataT> normalized() const;

    RMAGINE_INLINE_FUNCTION
    void normalize();

    RMAGINE_INLINE_FUNCTION
    void setZeros();

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator+(const Vector3_<DataT>& b) const
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
        return negation();
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

    ////////////////////////
    // CASTING

    template<typename ConvT>
    RMAGINE_INLINE_FUNCTION
    Vector3_<ConvT> cast() const
    {
        return {
            static_cast<ConvT>(x),
            static_cast<ConvT>(y),
            static_cast<ConvT>(z)
        };
    } 
};

} // namespace rmagine

#include "Vector3.tcc"

#endif // RMAGINE_MATH_VECTOR3_HPP
