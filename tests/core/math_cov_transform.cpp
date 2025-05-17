#include <iostream>

#include <rmagine/math/types.h>

#include <rmagine/math/memory_math.h>
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
bool equal(rm::Vector3_<DataT> a, rm::Vector3_<DataT> b)
{
    return ((a - b).l2norm() < 0.0001);
}

template<typename DataT>
bool equal_cov(
  const rm::Matrix_<DataT, 3, 3>& A,
  const rm::Matrix_<DataT, 3, 3>& B)
{
    rm::Matrix_<DataT, 3, 3> cov_diff = A - B;
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
bool equal(rm::CrossStatistics_<DataT> a, rm::CrossStatistics_<DataT> b)
{
    if(a.n_meas != b.n_meas)
    {
        return false;
    }

    if(!equal(a.dataset_mean, b.dataset_mean))
    {
        return false;
    }

    if(!equal(a.model_mean, b.model_mean))
    {
        return false;
    }

    if(!equal_cov(a.covariance, b.covariance))
    {
        return false;
    }
    
    return true;
}

void test_transform_means()
{
  std::cout << "---------------------------------" << std::endl;
  std::cout << "TEST TRANSFORM MEANS" << std::endl;

  rm::Transform T_s1_b = {
    .R = rm::EulerAngles{0.5, 1.0, 0.0},
    .t = {1.0, 2.0, 3.0}
  };

  // offset in base coords
  // means: T_b{t}_b{t-1} or T_b{dataset}_b{model}
  rm::Transform T_off_b = {
    .R = rm::EulerAngles{0.0, 0.0, 1.0},
    .t = {0.0, -5.0, 10.0}
  };

  // 
  // Spaß mit Transformationen - Space and Time
  // 
  //       model     T_off_s1 (?)   dataset
  //        (s1) <----------------- (s1)
  //         |                       |
  // T_s1_b  |                       | T_s1_b
  //         |                       |
  //         v                       v
  //        (b)  <----------------- (b) 
  //                  T_off_b   
  // 
  // T_off_s1? follow along the known path: 
  rm::Transform T_off_s1 = ~T_s1_b * T_off_b * T_s1_b;


  // fill in 's1' coords
  size_t n_points = 100;
  rm::Memory<rm::Vector3> dataset_s1_s1(n_points);
  for(size_t i=0; i<n_points; i++)
  {
      float p = static_cast<double>(i) / 100.0;
      rm::Vector3 d = {-p, p*10.f, p};
      dataset_s1_s1[i] = d;
  }
  rm::Vector3 mean_dataset_s1_s1 = rm::mean(dataset_s1_s1)[0];

  
  // transform to base coords
  rm::Memory<rm::Vector3> dataset_s1_b = rm::mult1xN(rm::make_view(T_s1_b), dataset_s1_s1);
  rm::Vector3 mean_dataset_s1_b = rm::mean(dataset_s1_b)[0];

  // apply offsets in respective coordinate systems
  rm::Memory<rm::Vector3> model_s1_s1  = rm::mult1xN(rm::make_view(T_off_s1), dataset_s1_s1);
  rm::Vector3 mean_model_s1_s1 = rm::mean(model_s1_s1)[0];
  
  rm::Memory<rm::Vector3> model_s1_b   = rm::mult1xN(rm::make_view(T_off_b),  dataset_s1_b);
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


  if(!equal(mean_dataset_s1_s1, mean_dataset_s1_s1_2))
  {
    RM_THROW(rm::Exception, "mean_dataset_s1_s1 != mean_dataset_s1_s1_2");
  }

  if(!equal(mean_dataset_s1_b, mean_dataset_s1_b_2))
  {
    RM_THROW(rm::Exception, "mean_dataset_s1_b != mean_dataset_s1_b_2");
  }

  if(!equal(mean_model_s1_s1, mean_model_s1_s1_2))
  {
    RM_THROW(rm::Exception, "mean_model_s1_s1 != mean_model_s1_s1_2");
  }

  if(!equal(mean_model_s1_b, mean_model_s1_b_2))
  {
    RM_THROW(rm::Exception, "mean_model_s1_b != mean_model_s1_b_2");
  }
}

void test_transform_covs()
{
  std::cout << "---------------------------------" << std::endl;
  std::cout << "TEST TRANSFORM COVARIANCES / CROSS STATISTICS" << std::endl;


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

  // setup:
  // sensor1 -> base
  // sensor2 -> base

  rm::Transform T_s1_b = {
    .R = rm::EulerAngles{0.5, 1.0, 0.0},
    .t = {1.0, 2.0, 3.0}
  };

  rm::Transform T_off_b = {
    .R = rm::EulerAngles{0.0, 0.0, 1.0},
    .t = {0.0, -5.0, 10.0}
  };

  // 
  // Spaß mit Transformationen - Space and Time
  // 
  //       model     T_off_s1 (?)   dataset
  //        (s1) <----------------- (s1)
  //         |                       |
  // T_s1_b  |                       | T_s1_b
  //         |                       |
  //         v                       v
  //        (b)  <----------------- (b) 
  //                  T_off_b   
  // 
  // T_off_s1? follow along the known path: 
  rm::Transform T_off_s1 = ~T_s1_b * T_off_b * T_s1_b;


  // fill points in sensor space "s1"
  size_t n_points = 100;
  rm::Memory<rm::Vector3> dataset_s1_s1(n_points);
  for(size_t i=0; i<n_points; i++)
  {
      float p = static_cast<double>(i) / 100.0;
      rm::Vector3 d = {-p, p*10.f, p};
      dataset_s1_s1[i] = d;
  }

  rm::UmeyamaReductionConstraints params;
  params.max_dist = 100000000.0;

  std::cout << std::endl;
  std::cout << "Experiment 1" << std::endl;
  rm::CrossStatistics stats_exp1;
  { // Experiment 1
    // 1. caluclate cross statistics in frame 's1'
    // 2. transform cross statistics from frame 's1' to frame 'b'
    rm::Memory<rm::Vector3> model_s1_s1 = rm::mult1xN(rm::make_view(T_off_s1), dataset_s1_s1);

    rm::CrossStatistics stats_s1_s1 = rm::statistics_p2p(rm::Transform::Identity(), 
        {.points = dataset_s1_s1},  // dataset
        {.points = model_s1_s1}, // model
        params);

    rm::CrossStatistics stats_s1_b = T_s1_b * stats_s1_s1;
    stats_exp1 = stats_s1_b;
  }

  std::cout << stats_exp1 << std::endl;
  
  std::cout << std::endl;
  std::cout << "Experiment 2:" << std::endl;
  rm::CrossStatistics stats_exp2;
  { // Experiment 2
    // 1. transform data + model from frame 's1' to frame 'b'
    // 2. calculate cross statistics in frame 'b'
    rm::Memory<rm::Vector3> dataset_s1_b = rm::mult1xN(rm::make_view(T_s1_b), dataset_s1_s1);
    rm::Memory<rm::Vector3> model_s1_b = rm::mult1xN(rm::make_view(T_off_b), dataset_s1_b);

    rm::CrossStatistics stats_s1_b = rm::statistics_p2p(rm::Transform::Identity(), 
        {.points = dataset_s1_b},  // dataset
        {.points = model_s1_b}, // model
        params);

    stats_exp2 = stats_s1_b;
  }
  std::cout << stats_exp2 << std::endl;

  if(!equal(stats_exp1, stats_exp2))
  {
    RM_THROW(rm::Exception, "stats_exp1 != stats_exp2");
  }
}


void test_transform_covs_merge()
{
  std::cout << "----------------------------------------------------------------------" << std::endl;
  std::cout << "TEST TRANSFORM COVARIANCES / CROSS STATISTICS. MULTI SENSOR AND MERGE" << std::endl;


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

  // setup:
  // sensor1 -> base
  // sensor2 -> base

  rm::Transform T_s1_b = {
    .R = rm::EulerAngles{0.5, 1.0, 0.0},
    .t = {1.0, 2.0, 3.0}
  };

  rm::Transform T_s2_b = {
    .R = rm::EulerAngles{0.5, 1.0, M_PI},
    .t = {-1.0, -2.0, 3.0}
  };

  rm::Transform T_off_b = {
    .R = rm::EulerAngles{0.0, 0.0, 1.0},
    .t = {0.0, -5.0, 10.0}
  };

  // 
  // Spaß mit Transformationen - Space and Time
  // 
  //       model     T_off_s1 (?)   dataset
  //        (s1) <----------------- (s1)
  //         |                       |
  // T_s1_b  |                       | T_s1_b
  //         |                       |
  //         v                       v
  //        (b)  <----------------- (b) 
  //                  T_off_b   
  // 
  // T_off_s1? follow along the known path: 
  rm::Transform T_off_s1 = ~T_s1_b * T_off_b * T_s1_b;
  // analog to s2
  rm::Transform T_off_s2 = ~T_s2_b * T_off_b * T_s2_b;

  // fill points in sensor spaces
  size_t n_points = 100;
  rm::Memory<rm::Vector3> dataset_s1_s1(n_points);
  rm::Memory<rm::Vector3> dataset_s2_s2(n_points);
  for(size_t i=0; i<n_points; i++)
  {
      float p = static_cast<double>(i) / 100.0;
      dataset_s1_s1[i] = {-p, p*10.f, p};
      dataset_s2_s2[i] = {fmod(p, 1.f), p, powf(p, 0.5)};
  }

  rm::UmeyamaReductionConstraints params;
  params.max_dist = 100000000.0;

  std::cout << std::endl;
  std::cout << "Experiment 1" << std::endl;
  rm::CrossStatistics stats_exp1;
  { // Experiment 1
    // 1. caluclate cross statistics in frame 's1' and 's2'
    // 2. transform cross statistics from frame 's1' and 's2' to frame 'b'
    // 3. merge cross statistics
    rm::Memory<rm::Vector3> model_s1_s1 = rm::mult1xN(rm::make_view(T_off_s1), dataset_s1_s1);
    rm::Memory<rm::Vector3> model_s2_s2 = rm::mult1xN(rm::make_view(T_off_s2), dataset_s2_s2);

    // 1.
    rm::CrossStatistics stats_s1_s1 = rm::statistics_p2p(rm::Transform::Identity(), 
        {.points = dataset_s1_s1},  // dataset
        {.points = model_s1_s1}, // model
        params);
    rm::CrossStatistics stats_s2_s2 = rm::statistics_p2p(rm::Transform::Identity(),
        {.points = dataset_s2_s2},
        {.points = model_s2_s2},
        params);
    
    // 2.
    rm::CrossStatistics stats_s1_b = T_s1_b * stats_s1_s1;
    rm::CrossStatistics stats_s2_b = T_s2_b * stats_s2_s2;

    // 3.
    rm::CrossStatistics stats_b_b = stats_s1_b + stats_s2_b;

    stats_exp1 = stats_b_b;
  }

  std::cout << stats_exp1 << std::endl;
  
  std::cout << std::endl;
  std::cout << "Experiment 2:" << std::endl;
  rm::CrossStatistics stats_exp2;
  { // Experiment 2
    // 1. transform data + model from frame 's1' to frame 'b'
    // 2. calculate cross statistics in frame 'b'
    // 3. merge cross statistics

    // 1.
    rm::Memory<rm::Vector3> dataset_s1_b = rm::mult1xN(rm::make_view(T_s1_b), dataset_s1_s1);
    rm::Memory<rm::Vector3> model_s1_b = rm::mult1xN(rm::make_view(T_off_b), dataset_s1_b);
    rm::Memory<rm::Vector3> dataset_s2_b = rm::mult1xN(rm::make_view(T_s2_b), dataset_s2_s2);
    rm::Memory<rm::Vector3> model_s2_b = rm::mult1xN(rm::make_view(T_off_b), dataset_s2_b);

    // 2.
    rm::CrossStatistics stats_s1_b = rm::statistics_p2p(rm::Transform::Identity(), 
        {.points = dataset_s1_b},  // dataset
        {.points = model_s1_b}, // model
        params);
    rm::CrossStatistics stats_s2_b = rm::statistics_p2p(rm::Transform::Identity(), 
        {.points = dataset_s2_b},  // dataset
        {.points = model_s2_b}, // model
        params);

    // 3.
    rm::CrossStatistics stats_b_b = stats_s1_b + stats_s2_b;

    stats_exp2 = stats_b_b;
  }
  std::cout << stats_exp2 << std::endl;

  if(!equal(stats_exp1, stats_exp2))
  {
    RM_THROW(rm::Exception, "stats_exp1 != stats_exp2");
  }
}

int main(int argc, char** argv)
{
    srand((unsigned int) time(0));

    std::cout << "STATS TRANSFORM TEST" << std::endl;

    // this shows the general structure of the tests limited to transforming means
    test_transform_means();

    // this shows the most simple case of transforming cross statistics including 
    // two means and one covariance between space and time frames
    test_transform_covs();

    // this is the final test
    // once it works it shows that every crossstatistics can freely transformed and merged
    // this e.g. means we can register arbitrary sensor configurations
    test_transform_covs_merge();

    return 0;
}

