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

    RMAGINE_INLINE_FUNCTION
    operator Quaternion_<DataT>() const 
    {
        std::cout << "Cast Euler to Quat" << std::endl;
        // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        // TODO: check, 
        // 1. test: correct
        // TODO:
        // - generic trigo functions
        const DataT cr = cosf(roll / 2.0f);
        const DataT sr = sinf(roll / 2.0f);
        const DataT cp = cosf(pitch / 2.0f);
        const DataT sp = sinf(pitch / 2.0f);
        const DataT cy = cosf(yaw / 2.0f);
        const DataT sy = sinf(yaw / 2.0f);

        std::cout << "Cast Euler to Quat - Done." << std::endl;
        return {
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        };
    }

    RMAGINE_INLINE_FUNCTION
    operator Matrix_<DataT, 3, 3>() const
    {
        Matrix_<DataT, 3, 3> M;

        // Wrong?
        // TODO check
        // 1. test: correct
        const DataT cA = cosf(roll);
        const DataT sA = sinf(roll);
        const DataT cB = cosf(pitch);
        const DataT sB = sinf(pitch);
        const DataT cC = cosf(yaw);
        const DataT sC = sinf(yaw);

        M(0,0) =  cB * cC;
        M(0,1) = -cB * sC;
        M(0,2) =  sB;
    
        M(1,0) =  sA * sB * cC + cA * sC;
        M(1,1) = -sA * sB * sC + cA * cC;
        M(1,2) = -sA * cB;
        
        M(2,0) = -cA * sB * cC + sA * sC;
        M(2,1) =  cA * sB * sC + sA * cC;
        M(2,2) =  cA * cB;

        return M;
    }


    template<typename ConvT>
    EulerAngles_<ConvT> cast() const
    {
        return {
            static_cast<ConvT>(roll),
            static_cast<ConvT>(pitch),
            static_cast<ConvT>(yaw)
        };
    }
};

} // rmagine

#include "EulerAngles.tcc"

#endif // RMAGINE_MATH_EULER_ANGLES_HPPs