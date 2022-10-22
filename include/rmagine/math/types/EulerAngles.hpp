#ifndef RMAGINE_MATH_EULER_ANGLES_HPP
#define RMAGINE_MATH_EULER_ANGLES_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>
#include <initializer_list>
#include <iostream>

namespace rmagine
{

/**
 * @brief EulerAngles type
 * 
 */
template<typename DataT>
struct EulerAngles_ 
{
    // DATA
    DataT roll;     // x-axis
    DataT pitch;    // y-axis
    DataT yaw;      // z-axis


    // Functions
    RMAGINE_FUNCTION
    static EulerAngles_<DataT> Identity()
    {
        EulerAngles_<DataT> ret;
        ret.setIdentity();
        return ret;
    }

    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    RMAGINE_INLINE_FUNCTION
    void set(const Quaternion_<DataT>& q);

    RMAGINE_INLINE_FUNCTION
    void set(const Matrix_<DataT, 3, 3>& M);

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> mult(const Vector3_<DataT>& v) const;

    //////////////////
    // Operators

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator*(const Vector3_<DataT>& v) const 
    {
        return mult(v);
    }

    ///////////////
    // CASTING

    /**
     * @brief EulerAngles -> Quaternion
     * 
     * @return Quaternion_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    operator Quaternion_<DataT>() const;

    /**
     * @brief EulerAngles -> Rotation Matrix
     * 
     * @return Matrix_<DataT, 3, 3> 
     */
    RMAGINE_INLINE_FUNCTION
    operator Matrix_<DataT, 3, 3>() const;

    /**
     * @brief Data Type Cast to ConvT
     * 
     * @tparam ConvT 
     */
    template<typename ConvT>
    RMAGINE_INLINE_FUNCTION
    EulerAngles_<ConvT> cast() const;
};

} // rmagine

#include "EulerAngles.tcc"

#endif // RMAGINE_MATH_EULER_ANGLES_HPPs