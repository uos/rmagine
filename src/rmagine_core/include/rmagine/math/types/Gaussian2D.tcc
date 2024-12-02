#include "Gaussian2D.hpp"

namespace rmagine {

template<typename DataT> 
RMAGINE_INLINE_FUNCTION
Gaussian2D_<DataT> Gaussian2D_<DataT>::add(const Gaussian2D_<DataT>& o) const
{
    Gaussian2D_<DataT> ret;
    ret.n_meas = n_meas + o.n_meas;
    
    const DataT w1 = static_cast<DataT>(n_meas) / static_cast<DataT>(ret.n_meas);
    const DataT w2 = static_cast<DataT>(o.n_meas) / static_cast<DataT>(ret.n_meas);
    ret.mean = mean * w1 + o.mean * w2;

    const Matrix_<DataT,2,2> P1 = (mean - ret.mean).multT(mean - ret.mean);
    const Matrix_<DataT,2,2> P2 = (o.mean - ret.mean).multT(o.mean - ret.mean);
    ret.sigma = (sigma + P1) * w1 + (o.sigma + P2) * w2;

    return ret;
}

} // namespace rmagine