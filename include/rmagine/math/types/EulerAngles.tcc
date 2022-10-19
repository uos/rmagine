#include "EulerAngles.hpp"

namespace rmagine
{

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void EulerAngles_<DataT>::setIdentity()
{
    roll = 0.0;
    pitch = 0.0;
    yaw = 0.0;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void EulerAngles_<DataT>::set(const Quaternion_<DataT>& q)
{
    *this = static_cast<EulerAngles_<DataT> >(q);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void EulerAngles_<DataT>::set(const Matrix_<DataT, 3, 3>& M)
{
    // extracted from knowledge of Matrix3x3::set(EulerAngles)
    // plus EulerAngles::set(Quaternion)
    // TODO: check. tested once: correct
    
    // roll (x-axis)
    const DataT sinr_cosp = -M(1,2);
    const DataT cosr_cosp = M(2,2);
    
    // pitch (y-axis)
    const DataT sinp = M(0,2);

    // yaw (z-axis)
    const DataT siny_cosp = -M(0,1);
    const DataT cosy_cosp = M(0,0);

    // roll (x-axis)
    roll = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis)
    if (fabs(sinp) >= 1.0f)
    {
        pitch = copysignf(M_PI / 2, sinp); // use 90 degrees if out of range
    } else {
        pitch = asinf(sinp);
    }

    // yaw (z-axis)
    yaw = atan2f(siny_cosp, cosy_cosp);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> EulerAngles_<DataT>::mult(const Vector3_<DataT>& v) const
{
    Quaternion_<DataT> q;
    q = *this;
    return q * v;
}

} // namespace rmagine