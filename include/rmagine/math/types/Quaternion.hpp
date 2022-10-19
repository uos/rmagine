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
     * @brief Rotate a vector with this quaternion
     * 
     * @param p 
     * @return Vector 
     */
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> mult(const Vector3_<DataT>& p) const;

    RMAGINE_INLINE_FUNCTION
    DataT dot(const Quaternion_<DataT>& q) const;

    RMAGINE_INLINE_FUNCTION
    DataT l2normSquared() const;

    RMAGINE_INLINE_FUNCTION
    DataT l2norm() const;

    RMAGINE_INLINE_FUNCTION
    void normalize();

    RMAGINE_INLINE_FUNCTION
    void set(const Matrix_<DataT, 3, 3>& M);

    RMAGINE_INLINE_FUNCTION
    void set(const EulerAngles_<DataT>& e);

    // TODO: Quatenrion from rotation around an axis v by an angle a
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

    RMAGINE_INLINE_FUNCTION
    operator EulerAngles_<DataT>() const 
    {
        // TODO: check
        // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        // checked once

        // roll (x-axis)
        const DataT sinr_cosp = 2.0 * (w * x + y * z);
        const DataT cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        // pitch (y-axis)
        const DataT sinp = 2.0 * (w * y - z * x);
        // yaw (z-axis)
        const DataT siny_cosp = 2.0 * (w * z + x * y);
        const DataT cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        constexpr DataT PI_HALF = M_PI / 2.0;

        EulerAngles_<DataT> e;

        // roll (x-axis)
        e.roll = atan2f(sinr_cosp, cosr_cosp);

        // pitch (y-axis)
        if (fabs(sinp) >= 1.0f)
        {
            e.pitch = copysignf(PI_HALF, sinp); // use 90 degrees if out of range
        } else {
            e.pitch = asinf(sinp);
        }

        // yaw (z-axis)
        e.yaw = atan2f(siny_cosp, cosy_cosp);

        return e;
    }

    RMAGINE_INLINE_FUNCTION
    operator Matrix_<DataT, 3, 3>() const
    {
        Matrix_<DataT, 3, 3> res;
        res(0,0) = 2.0 * (w * w + x * x) - 1.0;
        res(0,1) = 2.0 * (x * y - w * z);
        res(0,2) = 2.0 * (x * z + w * y);
        res(1,0) = 2.0 * (x * y + w * z);
        res(1,1) = 2.0 * (w * w + y * y) - 1.0;
        res(1,2) = 2.0 * (y * z - w * x);
        res(2,0) = 2.0 * (x * z - w * y);
        res(2,1) = 2.0 * (y * z + w * x);
        res(2,2) = 2.0 * (w * w + z * z) - 1.0;
        return res;
    } 

    template<typename ConvT>
    Quaternion_<ConvT> cast() const
    {
        return {
            static_cast<ConvT>(x),
            static_cast<ConvT>(y),
            static_cast<ConvT>(z),
            static_cast<ConvT>(w)
        };
    }
};

} // namespace rmagine

#include "Quaternion.tcc"

#endif // RMAGINE_MATH_QUATERNION_HPP