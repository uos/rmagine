#include "Transform.hpp"

namespace rmagine
{

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Transform_<DataT>::setIdentity()
{
    R.setIdentity();
    t.setZeros();
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Transform_<DataT>::set(const Matrix_<DataT, 4, 4>& M)
{
    R = M.rotation();
    t = M.translation();
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Transform_<DataT> Transform_<DataT>::inv() const
{
    Transform_<DataT> Tinv;
    Tinv.R = R.inv();
    Tinv.t = -(Tinv.R * t);
    return Tinv;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Transform_<DataT> Transform_<DataT>::mult(const Transform_<DataT>& T2) const
{
    // P_ = R1 * (R2 * P + t2) + t1;
    Transform_<DataT> T3;
    T3.t = R * T2.t;
    T3.R = R * T2.R;
    T3.t += t;
    return T3;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
void Transform_<DataT>::multInplace(const Transform_<DataT>& T2)
{
    // P_ = R1 * (R2 * P + t2) + t1;
    // P_ = R1 * R2 * P + R1 * t2 + t1
    // =>
    // t_ = R1 * t2 + t1
    // R_ = R1 * R2
    t = R * T2.t + t;
    R = R * T2.R;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Vector3_<DataT> Transform_<DataT>::mult(const Vector3_<DataT>& v) const
{
    return R * v + t;
}

} // namespace rmagine