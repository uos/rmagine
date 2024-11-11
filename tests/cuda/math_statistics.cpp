
#include <rmagine/math/statistics.h>
#include <rmagine/math/statistics.cuh>

#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/prints.h>


namespace rm = rmagine;

// size_t n_points = 1000;
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


void test1()
{
  rm::StopWatch sw;
  double el;

  rm::Memory<rm::Vector3> dataset_points(n_points);
  rm::Memory<rm::Vector3> model_points(n_points);
  rm::Memory<unsigned int> dataset_mask(n_points);
  rm::Memory<unsigned int> dataset_ids(n_points);
  
  for(size_t i=0; i<n_points; i++)
  {
      float p = static_cast<double>(i) / static_cast<double>(n_points);
      rm::Vector3 d = {-p, p*10.f, p};
      rm::Vector3 m = {p, p, -p};

      dataset_points[i] = d;
      model_points[i] = m;

      dataset_mask[i] = i%2;
      dataset_ids[i] = i%4; // 0,1,2,3
      // dataset_ids[i] = 0;
  }


  std::cout << "Define dataset" << std::endl;

  rm::Memory<rm::Vector3, rm::VRAM_CUDA> dataset_points_ = dataset_points;
  rm::Memory<rm::Vector3, rm::VRAM_CUDA> model_points_ = model_points;

  rm::PointCloudView_<rm::VRAM_CUDA> dataset = {.points = dataset_points_};

  rm::PointCloudView_<rm::VRAM_CUDA> model = {.points = model_points_};


  rm::Transform Tpre = rm::Transform::Identity();

  rm::UmeyamaReductionConstraints params;
  params.max_dist = 20000.0;

  rm::CrossStatistics stats;


  sw();
  stats = rm::statistics_p2p(Tpre, dataset, model, params);
  el = sw();

  printStats(stats);

  std::cout << "Runtime: " << el << " s" << std::endl;



}


int main(int argc, char** argv)
{
  test1();
  



  return 0;
}
