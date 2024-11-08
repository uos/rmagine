#include "CrossStatistics.hpp"

namespace rmagine {



template<typename DataT> 
RMAGINE_INLINE_FUNCTION
CrossStatistics_<DataT> CrossStatistics_<DataT>::add(const CrossStatistics_<DataT>& o) const
{
    CrossStatistics_<DataT> ret;
    ret.n_meas = n_meas + o.n_meas;
    
    const DataT w1 = static_cast<double>(n_meas) / static_cast<double>(ret.n_meas);
    const DataT w2 = static_cast<double>(o.n_meas) / static_cast<double>(ret.n_meas);

    ret.dataset_mean = dataset_mean * w1 + o.dataset_mean * w2;
    ret.model_mean = model_mean * w1 + o.model_mean * w2;

    const Matrix_<DataT, 3,3> P1 = covariance * w1 + o.covariance * w2;
    const Matrix_<DataT, 3,3> P2 = (model_mean - ret.model_mean).multT(dataset_mean - ret.dataset_mean) * w1 
                                 + (o.model_mean - ret.model_mean).multT(o.dataset_mean - ret.dataset_mean) * w2;
    ret.covariance = P1 + P2;
    return ret;
}

template<typename DataT> 
RMAGINE_INLINE_FUNCTION
void CrossStatistics_<DataT>::addInplace(const CrossStatistics_<DataT>& o)
{
    const unsigned int n_meas_new = n_meas + o.n_meas;
    
    // numerical critical
    const DataT w1 = static_cast<double>(n_meas) / static_cast<double>(n_meas_new);
    const DataT w2 = 1.0 - w1;
    // const DataT w2 = static_cast<double>(o.n_meas) / static_cast<double>(n_meas_new);
    
    const Vector3_<DataT> dataset_mean_new = dataset_mean * w1 + o.dataset_mean * w2;
    const Vector3_<DataT> model_mean_new   = model_mean * w1 + o.model_mean * w2;

    const Matrix_<DataT, 3,3> P1 = covariance * w1 + o.covariance * w2;
    const Matrix_<DataT, 3,3> P2 = (model_mean - model_mean_new).multT(dataset_mean - dataset_mean_new) * w1 
                                 + (o.model_mean - model_mean_new).multT(o.dataset_mean - dataset_mean_new) * w2;

    model_mean = model_mean_new;
    dataset_mean = dataset_mean_new;
    covariance = P1 + P2;
    n_meas = n_meas_new;
}

template<typename DataT> 
RMAGINE_INLINE_FUNCTION
void CrossStatistics_<DataT>::addInplace2(const CrossStatistics_<DataT>& o)
{
    const unsigned int n_meas_new = n_meas + o.n_meas;
    
    const double w1 = static_cast<double>(n_meas) / static_cast<double>(n_meas_new);
    
    const double w2 = 1.0 - w1;

    const DataT alpha = static_cast<double>(o.n_meas) / static_cast<double>(n_meas_new);
    
    // // const DataT w2 = static_cast<double>(o.n_meas) / static_cast<double>(n_meas_new);

    const Vector3_<DataT> dataset_mean_diff = (o.dataset_mean - dataset_mean);
    const Vector3_<DataT> model_mean_diff = (o.model_mean - model_mean);



    // const Vector3_<DataT> dataset_mean_new = dataset_mean + dataset_mean_diff * alpha;
    // const Vector3_<DataT> model_mean_new = model_mean + model_mean_diff * alpha;




    const Vector3_<DataT> dataset_mean_new = dataset_mean * w1 + o.dataset_mean * w2;
    const Vector3_<DataT> model_mean_new = model_mean * w1 + o.model_mean * w2;



    const Matrix_<DataT, 3,3> P1 = covariance * w1 + o.covariance * w2;
    const Matrix_<DataT, 3,3> P2 = (model_mean - model_mean_new).multT(dataset_mean - dataset_mean_new) * w1 
                                 + (o.model_mean - model_mean_new).multT(o.dataset_mean - dataset_mean_new) * w2;

    model_mean = model_mean_new;
    dataset_mean = dataset_mean_new;
    covariance = P1 + P2;
    n_meas = n_meas_new;
}



} // namespace rmagine