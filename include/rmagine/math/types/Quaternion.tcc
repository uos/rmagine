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
    // https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    // TODO: test
    // 1. test: correct
    DataT tr = M.trace();

    if (tr > 0) { 
        const DataT S = sqrtf(tr + 1.0) * 2; // S=4*qw 
        w = 0.25f * S;
        x = (M(2,1) - M(1,2)) / S;
        y = (M(0,2) - M(2,0)) / S; 
        z = (M(1,0) - M(0,1)) / S; 
    } else if ((M(0,0) > M(1,1)) && (M(0,0) > M(2,2))) { 
        const DataT S = sqrtf(1.0 + M(0,0) - M(1,1) - M(2,2)) * 2.0f; // S=4*qx 
        w = (M(2,1) - M(1,2)) / S;
        x = 0.25f * S;
        y = (M(0,1) + M(1,0)) / S; 
        z = (M(0,2) + M(2,0)) / S; 
    } else if (M(1,1) > M(2,2) ) { 
        const DataT S = sqrtf(1.0 + M(1,1) - M(0,0) - M(2,2)) * 2.0f; // S=4*qy
        w = (M(0,2) - M(2,0)) / S;
        x = (M(0,1) + M(1,0)) / S; 
        y = 0.25f * S;
        z = (M(1,2) + M(2,1)) / S; 
    } else { 
        const DataT S = sqrtf(1.0 + M(2,2) - M(0,0) - M(1,1)) * 2.0f; // S=4*qz
        w = (M(1,0) - M(0,1)) / S;
        x = (M(0,2) + M(2,0)) / S;
        y = (M(1,2) + M(2,1)) / S;
        z = 0.25f * S;
    }
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Quaternion_<DataT>::set(const EulerAngles_<DataT>& e)
{
    // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    // TODO: check, 
    // 1. test: correct
    const DataT cr = cosf(e.roll / 2.0f);
    const DataT sr = sinf(e.roll / 2.0f);
    const DataT cp = cosf(e.pitch / 2.0f);
    const DataT sp = sinf(e.pitch / 2.0f);
    const DataT cy = cosf(e.yaw / 2.0f);
    const DataT sy = sinf(e.yaw / 2.0f);

    w = cr * cp * cy + sr * sp * sy;
    x = sr * cp * cy - cr * sp * sy;
    y = cr * sp * cy + sr * cp * sy;
    z = cr * cp * sy - sr * sp * cy;
}


} // namespace rmagine