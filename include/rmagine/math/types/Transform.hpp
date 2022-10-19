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

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    void operator=(const Matrix_<DataT, 4, 4>& M)
    {
        set(M);
    }

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