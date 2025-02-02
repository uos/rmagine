#include "CrossStatistics.hpp"

namespace rmagine {



template<typename DataT> 
RMAGINE_INLINE_FUNCTION
CrossStatistics_<DataT> CrossStatistics_<DataT>::add(const CrossStatistics_<DataT>& o) const
{
  CrossStatistics_<DataT> ret;
  ret.n_meas = n_meas + o.n_meas;

  if(ret.n_meas == 0)
  {
    return *this;
  }
  
  const DataT w1 = static_cast<DataT>(n_meas) / static_cast<DataT>(ret.n_meas);
  const DataT w2 = static_cast<DataT>(o.n_meas) / static_cast<DataT>(ret.n_meas);

  ret.dataset_mean = dataset_mean * w1 + o.dataset_mean * w2;
  ret.model_mean = model_mean * w1 + o.model_mean * w2;

  const Matrix_<DataT,3,3> P1 = (model_mean - ret.model_mean).multT(dataset_mean - ret.dataset_mean);
  const Matrix_<DataT,3,3> P2 = (o.model_mean - ret.model_mean).multT(o.dataset_mean - ret.dataset_mean);
  
  ret.covariance = (covariance + P1) * w1 + (o.covariance + P2) * w2;
  
  return ret;
}

template<typename DataT> 
RMAGINE_INLINE_FUNCTION
void CrossStatistics_<DataT>::addInplace(const CrossStatistics_<DataT>& o)
{
  const unsigned int n_meas_new = n_meas + o.n_meas;
  
  if(n_meas_new == 0)
  {
    return;
  }

  // numerical critical
  const DataT w1 = static_cast<DataT>(n_meas) / static_cast<DataT>(n_meas_new);
  const DataT w2 = static_cast<DataT>(o.n_meas) / static_cast<DataT>(n_meas_new);
  
  const Vector3_<DataT> dataset_mean_new = dataset_mean * w1 + o.dataset_mean * w2;
  const Vector3_<DataT> model_mean_new   = model_mean * w1 + o.model_mean * w2;

  const Matrix_<DataT,3,3> P1 = (model_mean - model_mean_new).multT(dataset_mean - dataset_mean_new);
  const Matrix_<DataT,3,3> P2 = (o.model_mean - model_mean_new).multT(o.dataset_mean - dataset_mean_new);

  model_mean = model_mean_new;
  dataset_mean = dataset_mean_new;
  covariance = (covariance + P1) * w1 + (o.covariance + P2) * w2;
  n_meas = n_meas_new;
}

template<typename DataT> 
RMAGINE_INLINE_FUNCTION
void CrossStatistics_<DataT>::addInplace(volatile CrossStatistics_<DataT>& o) volatile
{
    const unsigned int n_meas_new = n_meas + o.n_meas;
    
    // numerical critical
    volatile DataT w1 = static_cast<DataT>(n_meas) / static_cast<DataT>(n_meas_new);
    volatile DataT w2 = static_cast<DataT>(o.n_meas) / static_cast<DataT>(n_meas_new);
    
    throw std::runtime_error("NOT IMPLEMENTED");

    // volatile Vector3_<DataT> dataset_mean_new = dataset_mean * w1 + o.dataset_mean * w2;
    // volatile Vector3_<DataT> model_mean_new   = model_mean * w1 + o.model_mean * w2;

    // volatile Matrix_<DataT, 3,3> P1 = covariance * w1 + o.covariance * w2;
    // volatile Matrix_<DataT, 3,3> P2 = (model_mean - model_mean_new).multT(dataset_mean - dataset_mean_new) * w1 
    //                              + (o.model_mean - model_mean_new).multT(o.dataset_mean - dataset_mean_new) * w2;

    // model_mean.x = model_mean_new.x;
    // model_mean = model_mean_new;
    // dataset_mean = dataset_mean_new;
    // covariance = P1 + P2;
    // n_meas = n_meas_new;
}

} // namespace rmagine