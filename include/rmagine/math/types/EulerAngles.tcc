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
    *this = static_cast<EulerAngles_<DataT> >(M);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> EulerAngles_<DataT>::mult(const Vector3_<DataT>& v) const
{
    Quaternion_<DataT> q;
    q = *this;
    return q * v;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
EulerAngles_<DataT>::operator Quaternion_<DataT>() const 
{
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

template<typename DataT>
RMAGINE_INLINE_FUNCTION
EulerAngles_<DataT>::operator Matrix_<DataT, 3, 3>() const
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

template<typename DataT>
template<typename ConvT> 
RMAGINE_INLINE_FUNCTION 
EulerAngles_<ConvT> EulerAngles_<DataT>::cast() const
{
    return {
        static_cast<ConvT>(roll),
        static_cast<ConvT>(pitch),
        static_cast<ConvT>(yaw)
    };
}

} // namespace rmagine