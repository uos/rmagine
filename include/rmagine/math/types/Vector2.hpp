#ifndef RMAGINE_MATH_VECTOR2_HPP
#define RMAGINE_MATH_VECTOR2_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>

namespace rmagine
{

template<typename DataT>
struct Vector2_
{
    DataT x;
    DataT y;

    // FUNCTIONS
    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> add(const Vector2_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    void addInplace(const Vector2_<DataT>& b);

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> sub(const Vector2_<DataT>& b) const;

    RMAGINE_INLINE_FUNCTION
    void subInplace(const Vector2_<DataT>& b);

    RMAGINE_INLINE_FUNCTION
    Vector2_<DataT> negation() const;

    RMAGINE_INLINE_FUNCTION
    void negate();

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
    Vector2_<DataT> div(const DataT& s) const;

    RMAGINE_INLINE_FUNCTION
    void divInplace(const DataT& s);

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
        return negation();
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

    RMAGINE_INLINE_FUNCTION
    void operator=(const Matrix_<DataT, 2, 1>& M)
    {
        x = M(0, 0);
        y = M(1, 0);
    }
};


} // namespace rmagine

#include "Vector2.tcc"

#endif // RMAGINE_MATH_VECTOR2_HPP