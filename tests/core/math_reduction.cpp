#include <iostream>


#include <rmagine/math/types.h>

#include <rmagine/math/memory_math.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <rmagine/math/statistics.h>

#include <cassert>

#include <algorithm>
#include <random>


#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>


namespace rm = rmagine;

size_t n_points = 10000000;

template<typename T>
bool is_valid(T a)
{
    return a == a;
}

template<typename DataT>
void printStats(rm::CrossStatistics_<DataT> stats)
{
  std::cout << "CrossStatistics: " << std::endl;
  std::cout << "- dataset mean: " << stats.dataset_mean << std::endl;
  std::cout << "- model mean: " << stats.model_mean << std::endl;
  std::cout << "- cov: " << stats.covariance << std::endl;
  std::cout << "- n meas: " << stats.n_meas << std::endl; 
}

template<typename DataT>
void checkStats(rm::CrossStatistics_<DataT> stats)
{
  // check for nans
  if(!is_valid(stats.dataset_mean.x)){throw std::runtime_error("ERROR: NAN");};
  if(!is_valid(stats.dataset_mean.y)){throw std::runtime_error("ERROR: NAN");};
  if(!is_valid(stats.dataset_mean.z)){throw std::runtime_error("ERROR: NAN");};
  
  if(!is_valid(stats.model_mean.x)){throw std::runtime_error("ERROR: NAN");};
  if(!is_valid(stats.model_mean.y)){throw std::runtime_error("ERROR: NAN");};
  if(!is_valid(stats.model_mean.z)){throw std::runtime_error("ERROR: NAN");};

  if(!is_valid(stats.covariance(0,0))){throw std::runtime_error("ERROR: NAN");};
  if(!is_valid(stats.covariance(0,1))){throw std::runtime_error("ERROR: NAN");};
  if(!is_valid(stats.covariance(0,2))){throw std::runtime_error("ERROR: NAN");};

  if(!is_valid(stats.covariance(1,0))){throw std::runtime_error("ERROR: NAN");};
  if(!is_valid(stats.covariance(1,1))){throw std::runtime_error("ERROR: NAN");};
  if(!is_valid(stats.covariance(1,2))){throw std::runtime_error("ERROR: NAN");};

  if(!is_valid(stats.covariance(2,0))){throw std::runtime_error("ERROR: NAN");};
  if(!is_valid(stats.covariance(2,1))){throw std::runtime_error("ERROR: NAN");};
  if(!is_valid(stats.covariance(2,2))){throw std::runtime_error("ERROR: NAN");};
}

template<typename DataT>
bool equal(rm::CrossStatistics_<DataT> a, rm::CrossStatistics_<DataT> b)
{
  if(a.n_meas != b.n_meas)
  {
    return false;
  }

  if((a.dataset_mean - b.dataset_mean).l2norm() > 0.0001)
  {
    return false;
  }

  if((a.model_mean - b.model_mean).l2norm() > 0.0001)
  {
    return false;
  }

  rm::Matrix_<DataT, 3, 3> cov_diff = a.covariance - b.covariance;
  double cov_diff_abs_sum = 0.0;

  for(size_t i=0; i<3; i++)
  {
    for(size_t j=0; j<3; j++)
    {
      cov_diff_abs_sum += fabs(cov_diff(i, j));
    }
  }
  
  cov_diff_abs_sum /= 9.0;

  if(cov_diff_abs_sum > 0.0001)
  {
    return false;
  }
  
  return true;
}

template<typename DataT>
void omp_vs_std_reduction()
{
  rm::StopWatch sw;
  double el;

  // fill
  std::vector<rm::Vector3_<DataT> > points(n_points);
  for(size_t i=0; i<n_points; i++)
  {
    DataT p = static_cast<double>(i) / static_cast<double>(n_points);
    rm::Vector3_<DataT> d = {-p, p*10.f, p};
    points[i] = d;
  }

  // 1. sequential

  rm::Vector3_<DataT> D1 = {0.0, 0.0, 0.0};

  sw();
  for(size_t i=0; i<n_points; i++)
  {
      DataT p = static_cast<double>(i) / static_cast<double>(n_points);
      rm::Vector3_<DataT> d = {-p, p*10.f, p};
      D1 += d;
  }
  D1 /= static_cast<DataT>(n_points);

  el = sw();

  std::cout << "Sequential Reduction:" << std::endl;
  std::cout << "- res: " << D1 << std::endl;
  std::cout << "- runtime: " << el << " s" << std::endl;


  rm::Vector3_<DataT> D2 = {0.0, 0.0, 0.0};

  sw();

  D2 = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, points.size()),
    rm::Vector3_<DataT>{0.0, 0.0, 0.0},
    [&](const tbb::blocked_range<size_t>& r, rm::Vector3_<DataT> acc) 
    {
      for (size_t i = r.begin(); i != r.end(); ++i) 
      {
        acc += points[i];
      }
      return acc;
    },
    std::plus<rm::Vector3_<DataT> >()
  );

  D2 /= static_cast<DataT>(n_points);
  el = sw();

  std::cout << "TBB Reduction:" << std::endl;
  std::cout << "- res: " << D2 << std::endl;
  std::cout << "- runtime: " << el << " s" << std::endl;

  // I still have to figure out:
  // - how I can use only OpenMP as g++ parallelization backend 
  //    (even with -fopenmp we get linker errors to missing tbb)
  // - otherwise, we get a hard dependency to tbb. Do we want that?

  // sw();
  // rm::Vector3_<DataT> D3 = std::reduce(std::execution::seq, points.begin(), points.end());
  // D3 /= static_cast<DataT>(n_points);
  // el = sw();

  // std::cout << "std::reduce, sequential:" << std::endl;
  // std::cout << "- res: " << D3 << std::endl;
  // std::cout << "- runtime: " << el << std::endl;

  // sw();
  // rm::Vector3_<DataT> D4 = std::reduce(std::execution::par, points.begin(), points.end());
  // D4 /= static_cast<DataT>(n_points);
  // el = sw();

  // std::cout << "std::reduce, parallel:" << std::endl;
  // std::cout << "- res: " << D4 << std::endl;
  // std::cout << "- runtime: " << el << std::endl;
}


int main(int argc, char** argv)
{
  std::cout << "REDUCTION TEST" << std::endl;

  std::cout << "First Run" << std::endl;
  omp_vs_std_reduction<double>();

  std::cout << "Second Run" << std::endl;
  omp_vs_std_reduction<double>();

  return 0;
}

