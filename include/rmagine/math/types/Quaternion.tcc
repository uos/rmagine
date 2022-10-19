#include "Quaternion.hpp"
#include <math.h>

namespace rmagine
{


template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Quaternion_<DataT>::setIdentity()
{
    x = 0.0;
    y = 0.0;
    z = 0.0;
    w = 1.0;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Quaternion_<DataT> Quaternion_<DataT>::inv() const 
{
    return {-x, -y, -z, w};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Quaternion_<DataT>::invInplace()
{
    x = -x;
    y = -y;
    z = -z;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Quaternion_<DataT> Quaternion_<DataT>::mult(const Quaternion_<DataT>& q2) const 
{
    return {w*q2.x + x*q2.w + y*q2.z - z*q2.y,
            w*q2.y - x*q2.z + y*q2.w + z*q2.x,
            w*q2.z + x*q2.y - y*q2.x + z*q2.w,
            w*q2.w - x*q2.x - y*q2.y - z*q2.z};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Quaternion_<DataT>::multInplace(const Quaternion_<DataT>& q2) 
{
    const Quaternion_<DataT> tmp = mult(q2);
    x = tmp.x;
    y = tmp.y;
    z = tmp.z;
    w = tmp.w;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> Quaternion_<DataT>::mult(const Vector3_<DataT>& p) const
{
    const Quaternion_<DataT> P{p.x, p.y, p.z, 0.0};
    const Quaternion_<DataT> PT = this->mult(P).mult(inv());
    return {PT.x, PT.y, PT.z};
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Quaternion_<DataT>::dot(const Quaternion_<DataT>& q) const
{
    return x * q.x + y * q.y + z * q.z + w * q.w;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Quaternion_<DataT>::l2normSquared() const 
{
    return w * w + x * x + y * y + z * z;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
DataT Quaternion_<DataT>::l2norm() const 
{
    return sqrtf(l2normSquared());
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Quaternion_<DataT>::normalize()
{
    const DataT d = l2norm();
    x /= d;
    y /= d;
    z /= d;
    w /= d;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Quaternion_<DataT>::set(const Matrix_<DataT, 3, 3>& M)
{
    *this = static_cast<Quaternion_<DataT> >(M);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Quaternion_<DataT>::set(const EulerAngles_<DataT>& e)
{
    *this = static_cast<Quaternion_<DataT> >(e);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Quaternion_<DataT>::operator EulerAngles_<DataT>() const 
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

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Quaternion_<DataT>::operator Matrix_<DataT, 3, 3>() const
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

template<typename DataT>
template<typename ConvT>
RMAGINE_INLINE_FUNCTION
Quaternion_<ConvT> Quaternion_<DataT>::cast() const
{
    return {
        static_cast<ConvT>(x),
        static_cast<ConvT>(y),
        static_cast<ConvT>(z),
        static_cast<ConvT>(w)
    };
}


} // namespace rmagine