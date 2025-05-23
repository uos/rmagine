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

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Gaussian3D_<DataT> Transform_<DataT>::mult(const Gaussian3D_<DataT>& g) const
{
    Gaussian3D_<DataT> res;
    res.mean = mult(g.mean);
    const Matrix_<DataT, 3, 3> M = R; // TODO: can we do the two steps more efficient?
    res.sigma = M * g.sigma * M.T();
    res.n_meas = g.n_meas;
    return res; 
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
CrossStatistics_<DataT> Transform_<DataT>::mult(const CrossStatistics_<DataT>& stats) const
{
    CrossStatistics_<DataT> res;
    res.dataset_mean = mult(stats.dataset_mean);
    res.model_mean = mult(stats.model_mean);
    const Matrix_<DataT, 3, 3> M = R; // TODO: can we do the two steps more efficient?
    res.covariance = M * stats.covariance * M.T();
    res.n_meas = stats.n_meas;
    return res; 
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Transform_<DataT> Transform_<DataT>::to(const Transform_<DataT>& T2) const
{
    return inv().mult(T2);
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Transform_<DataT> Transform_<DataT>::pow(const DataT& exp) const
{
    Transform_<DataT> res;
    res.R = R.pow(exp);
    res.t = t * exp;
    return res;
}

template<typename DataT>
RMAGINE_INLINE_FUNCTION
Transform_<DataT>::operator Matrix_<DataT, 4, 4>() const 
{
    Matrix_<DataT, 4, 4> M;
    M.set(*this);
    return M;
}

} // namespace rmagine