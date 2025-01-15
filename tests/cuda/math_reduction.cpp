
#include <rmagine/math/statistics.h>
#include <rmagine/math/statistics.cuh>

#include <rmagine/math/math.cuh>

#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/prints.h>


namespace rm = rmagine;

// size_t n_points = 1000;
// size_t n_points = 1920*1080;
size_t n_points = 1023;

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
  
  for(size_t i=0; i < n_points; i++)
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
  
  std::cout << "WARMUP" << std::endl;
  auto _ = rm::statistics_p2p(Tpre, dataset, model, params);

  std::cout << "P2P Test 1" << std::endl;

  rm::Memory<rm::CrossStatistics, rm::VRAM_CUDA> stats_gpu(1);
  sw();
  rm::statistics_p2p(Tpre, dataset, model, params, stats_gpu);
  el = sw();

  rm::Memory<rm::CrossStatistics> stats = stats_gpu;

  printStats(stats[0]);

  std::cout << "Runtime: " << el << " s" << std::endl;

  // rm::Memory<rm::Vector, rm::VRAM_CUDA> dataset_points_sum(1);

  // sw();
  // rm::sum(dataset_points_, dataset_points_sum);
  // el = sw();

  // std::cout << "Sum: " << el << " s" << std::endl;
}

void preinit_cuda()
{
  int x = 1;
  rm::Memory<int, rm::VRAM_CUDA> bla = rm::MemoryView<int, rm::RAM>(&x, 1);

  rm::Memory<int, rm::RAM> y = rm::sum(bla);

  std::cout << y[0] << std::endl;
}

void test_sum_1()
{   
    rm::Memory<int> seq(32);
    for(int i=0; i<seq.size(); i++)
    {
      seq[i] = i+1;
    }
    
    std::cout << "Input:" << std::endl;
    for(size_t i=0; i<seq.size(); i++)
    {
      std::cout << seq[i] << " ";
    }
    std::cout << std::endl;

    // upload
    rm::Memory<int, rm::VRAM_CUDA> seq_ = seq;
    rm::Memory<int, rm::VRAM_CUDA> sums_;
    while(seq_.size() > 1)
    {
      sums_.resize(seq_.size() / 4);
      if(sums_.size() < 1)
      {
        sums_.resize(1);
      }
      rm::sum_reduce_test_t4(seq_, sums_);

      { // print
        rm::Memory<int> sums = sums_;
        std::cout << "Reduced to" << std::endl;
        for(size_t i=0; i<sums.size(); i++)
        {
          std::cout << sums[i] << " ";
        }
        std::cout << std::endl;
      }

      seq_ = sums_;
    }
}

void test_sum_2()
{
   
    // 3 blocks, 4 threads, 5 elements per thread to reduce initially
    rm::Memory<int> seq(3 * 4 * 5);
    for(int i=0; i<seq.size(); i++)
    {
      seq[i] = i+1;
    }
    
    std::cout << "Input:" << std::endl;
    for(size_t i=0; i<seq.size(); i++)
    {
      std::cout << seq[i] << " ";
    }
    std::cout << std::endl;

    // upload
    rm::Memory<int, rm::VRAM_CUDA> seq_ = seq;
    rm::Memory<int, rm::VRAM_CUDA> sums_(3);
    rm::sum_reduce_test_t4(seq_, sums_);

    { // print
      rm::Memory<int> sums = sums_;
      std::cout << "Reduced to" << std::endl;
      for(size_t i=0; i<sums.size(); i++)
      {
        std::cout << sums[i] << " ";
      }
      std::cout << std::endl;
    }
}

int main(int argc, char** argv)
{

  // preinit cuda
  preinit_cuda();
  

  // test_sum_1();

  // command to show what block 0 is doing
  // ./bin/rmagine_tests_cuda_math_reduction | grep "b0"
  //
  // command to show what thread 0 of block 0 is doing:
  // ./bin/rmagine_tests_cuda_math_reduction | grep "b0, t0"
  // 
  // command to show what block 0 is initially reducing from global memory to shared memory:
  // ./bin/rmagine_tests_cuda_math_reduction | grep "b0" | grep "data -> smem"
  test_sum_2();



  return 0;
}
