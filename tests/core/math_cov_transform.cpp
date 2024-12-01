#include <iostream>

#include <rmagine/math/types.h>

#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <rmagine/math/statistics.h>

#include <cassert>

#include <rmagine/math/omp.h>

#include <algorithm>
#include <random>

namespace rm = rmagine;

size_t n_points = 1000000;

template<typename T>
bool is_valid(T a)
{
    return a == a;
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

    if( (a.dataset_mean - b.dataset_mean).l2norm() > 0.0001)
    {
        return false;
    }

    if( (a.model_mean - b.model_mean).l2norm() > 0.0001)
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

rm::CrossStatistics operator*(rm::Transform T, rm::CrossStatistics stats)
{
  rm::CrossStatistics res;

  res.dataset_mean = T * stats.dataset_mean;
  res.model_mean = T * stats.model_mean;
  const rm::Matrix3x3 R = T.R;
  res.covariance = R * stats.covariance * R.T();
  res.n_meas = stats.n_meas;

  return res; 
}

void test_transform_1()
{
  rm::Transform T_s1_b = {
    .R = rm::EulerAngles{0.5, 1.0, 0.0},
    .t = {1.0, 2.0, 3.0}
  };

  // offset in base coords
  rm::Transform T_off_b = {
    .R = rm::EulerAngles{0.0, 0.0, 1.0},
    .t = {0.0, -5.0, 10.0}
  };


  size_t n_points = 100;
  rm::Memory<rm::Vector3> dataset_s1_s1(n_points);
  // fill in 's1' coords
  for(size_t i=0; i<n_points; i++)
  {
      float p = static_cast<double>(i) / 100.0;
      rm::Vector3 d = {-p, p*10.f, p};
      dataset_s1_s1[i] = d;
  }
  // transform to base coords
  rm::Memory<rm::Vector3> dataset_s1_b = rm::mult1xN(rm::make_view(T_s1_b),   dataset_s1_s1);

  // transform offset to base coords
  rm::Transform T_off_s1 = ~T_s1_b * T_off_b;

  // apply offsets in respective coordinate systems
  rm::Memory<rm::Vector3> model_s1_s1  = rm::mult1xN(rm::make_view(T_off_s1), dataset_s1_s1);
  rm::Memory<rm::Vector3> model_s1_b   = rm::mult1xN(rm::make_view(T_off_b),  dataset_s1_b);

  // calculate means of mode and dataset

  // direct way
  rm::Vector3 mean_dataset_s1_s1 = rm::mean(dataset_s1_s1)[0];
  rm::Vector3 mean_dataset_s1_b = rm::mean(dataset_s1_b)[0];

  rm::Vector3 mean_model_s1_s1 = rm::mean(model_s1_s1)[0];
  rm::Vector3 mean_model_s1_b = rm::mean(model_s1_b)[0];


  // transforming means instead
  rm::Vector3 mean_dataset_s1_s1_2 = ~T_s1_b * mean_dataset_s1_b;
  rm::Vector3 mean_dataset_s1_b_2  = T_s1_b * mean_dataset_s1_s1;

  rm::Vector3 mean_model_s1_s1_2 = ~T_s1_b * mean_model_s1_b;
  rm::Vector3 mean_model_s1_b_2  = T_s1_b * mean_model_s1_s1;


  std::cout << "Mean Dataset in 's1' coords: " << mean_dataset_s1_s1 << " == " << mean_dataset_s1_s1_2 << std::endl;
  std::cout << "Mean Dataset in  'b' coords: " << mean_dataset_s1_b << " == " << mean_dataset_s1_b_2 << std::endl;
  std::cout << "Mean Model   in 's1' coords: " << mean_model_s1_s1 << " == " << mean_model_s1_s1_2 << std::endl;
  std::cout << "Mean Model   in  'b' coords: " << mean_model_s1_b << " == " << mean_model_s1_b_2 << std::endl;
  



}

void test_transform_2()
{
  size_t n_points = 100;

  std::cout << "TEST TRANSFORM 1" << std::endl;
  std::cout << "- n_points: " << n_points << std::endl;

  // Test Setup:
  // given:
  // - dataset of sensor 1 + transform from sensor 1 to base
  // - transform from base to model
  // 
  // Experiment 1:
  // 1. calculate CrossStatistics in sensor1 coords
  // 2. Transform CrossStatistics to base coords
  // 
  // Experiment 2:
  // 1. transform dataset to base coords
  // 2. calculate CrossStatistics in base coords
  //
  // Aim: The resulting CrossStatistics of all experiments should be the same

  rm::Memory<rm::Vector3> dataset_s1_s1(n_points);
  
  // fill
  for(size_t i=0; i<n_points; i++)
  {
      float p = static_cast<double>(i) / 100.0;
      rm::Vector3 d = {-p, p*10.f, p};
      dataset_s1_s1[i] = d;
  }

  // setup:
  // sensor1 -> base
  // sensor2 -> base
  rm::Transform T_s1_b = {
    .R = rm::EulerAngles{0.5, 1.0, 0.0},
    .t = {1.0, 2.0, 3.0}
  };

  rm::Transform T_b_m = {
    .R = rm::EulerAngles{0.0, 0.0, 1.0},
    .t = {0.0, -5.0, 10.0}
  };


  rm::UmeyamaReductionConstraints params;
  params.max_dist = 100000000.0;

  std::cout << std::endl;
  std::cout << "Experiment 1" << std::endl;
  rm::CrossStatistics stats_exp1;
  { // Experiment 1
    rm::Transform T_s1_m = T_b_m * T_s1_b;
    rm::Memory<rm::Vector3> model_s1_s1 = rm::mult1xN(make_view(T_s1_m), dataset_s1_s1);

    rm::CrossStatistics stats_s1_s1 = rm::statistics_p2p(rm::Transform::Identity(), 
        {.points = dataset_s1_s1},  // dataset
        {.points = model_s1_s1}, // model
        params);
    
    // std::cout << "stats_s1_s1" << std::endl;
    // printStats(stats_s1_s1);

    rm::CrossStatistics stats_s1_b = T_s1_b * stats_s1_s1;
    
    stats_exp1 = stats_s1_b;
  }

  std::cout << stats_exp1 << std::endl;
  
  std::cout << std::endl;
  std::cout << "Experiment 2:" << std::endl;
  rm::CrossStatistics stats_exp2;
  { // Experiment 2
    rm::Memory<rm::Vector3> dataset_s1_b = rm::mult1xN(make_view(T_s1_b), dataset_s1_s1);
    rm::Memory<rm::Vector3> model_s1_b = rm::mult1xN(make_view(T_b_m), dataset_s1_b);

    rm::CrossStatistics stats_s1_b = rm::statistics_p2p(rm::Transform::Identity(), 
        {.points = dataset_s1_b},  // dataset
        {.points = model_s1_b}, // model
        params);

    // std::cout << "stats_s1_b" << std::endl;
    // printStats(stats_s1_b);

    stats_exp2 = stats_s1_b;
  }
  std::cout << stats_exp2 << std::endl;

  // std::cout << dataset_sensor1[100] << " -> " << model_points[100] << std::endl;

}


int main(int argc, char** argv)
{
    srand((unsigned int) time(0));

    std::cout << "STATS TEST" << std::endl;

    test_transform_1();
    // test_transform_2();

    return 0;
}

