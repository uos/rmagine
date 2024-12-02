#include "Gaussian1D.hpp"

namespace rmagine {

template<typename DataT> 
RMAGINE_INLINE_FUNCTION
Gaussian1D_<DataT> Gaussian1D_<DataT>::add(const Gaussian1D_<DataT>& o) const
{
    Gaussian1D_<DataT> ret;
    ret.n_meas = n_meas + o.n_meas;
    
    const DataT w1 = static_cast<DataT>(n_meas) / static_cast<DataT>(ret.n_meas);
    const DataT w2 = static_cast<DataT>(o.n_meas) / static_cast<DataT>(ret.n_meas);
    ret.mean = mean * w1 + o.mean * w2;

    const DataT P1 = (mean - ret.mean);
    const DataT P2 = (o.mean - ret.mean);
    ret.sigma = (sigma + P1*P1) * w1 + (o.sigma + P2*P2) * w2;
    
    return ret;
}

} // namespace rmagine