
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

void test_sum()
{
    rm::StopWatch sw;
    double el;

    rm::Memory<int> seq(1920 * 1080 * 10);
    // sames reducing 10 images of size 1920*1080
    // -- just for imagination. if you really want to reduce 10 images, you should use a batch implemention
    for(size_t i=0; i<seq.size(); i++)
    {
      seq[i] = 1;
    }
    // 1, 2, 3, 4, .. 16
    rm::Memory<int, rm::VRAM_CUDA> seq_gpu = seq;

    rm::Memory<int> seq_sum(1080);
    for(size_t i=0; i<seq_sum.size(); i++)
    {
      seq_sum[i] = 0;
    }

    // prepare extra memory
    rm::Memory<int, rm::VRAM_CUDA> seq_sum_gpu = seq_sum;
    rm::Memory<int, rm::VRAM_CUDA> total_gpu(1);

    std::cout << std::endl;
    std::cout << "Warmup" << std::endl;
    rm::sum(seq_gpu, total_gpu);

    std::cout << std::endl;
    std::cout << "TEST 1: Reduction in one block" << std::endl;
    rm::sum(seq_gpu, total_gpu);
    rm::Memory<int> total1 = total_gpu;

    std::cout << total1[0] << " == " << seq.size() << std::endl;

    std::cout << std::endl;
    std::cout << "TEST 2: Two-fold reduction. 1. Many blocks -> Result per block. 2. Reduce intermediate results per block." << std::endl;
    
    rm::sum(seq_gpu, seq_sum_gpu);
    // std::cout << "-" << std::endl;
    rm::sum(seq_sum_gpu, total_gpu);
    rm::Memory<int> total2 = total_gpu;
    std::cout << total2[0] << " == " << seq.size() << std::endl;
    
    // in my benchmarks, TEST 1 was still faster than TEST2
}


int main(int argc, char** argv)
{
  // n_points = 1024;


  // preinit cuda
  // preinit_cuda();
  

  // test1();

  // test2();

  test_sum();



  return 0;
}
