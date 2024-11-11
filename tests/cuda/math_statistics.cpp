
#include <rmagine/math/statistics.h>
#include <rmagine/math/statistics.cuh>

#include <rmagine/math/math.cuh>

#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/prints.h>


namespace rm = rmagine;

// size_t n_points = 1000;
size_t n_points = 1000*1000;

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
  
  for(size_t i=1; i <= n_points; i++)
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
  

  auto _ = rm::statistics_p2p(Tpre, dataset, model, params);

  rm::Memory<rm::CrossStatistics, rm::VRAM_CUDA> stats_gpu(1);
  sw();
  rm::statistics_p2p(Tpre, dataset, model, params, stats_gpu);
  el = sw();

  rm::Memory<rm::CrossStatistics> stats = stats_gpu;

  printStats(stats[0]);

  std::cout << "Runtime: " << el << " s" << std::endl;



}


void test2()
{
    size_t n_elements = n_points;
    rm::StopWatch sw;
    double el;

    rm::Memory<int, rm::RAM> seq(n_elements);
    for(int i=1; i <= n_elements; i++)
    {
      seq[i-1] = 1;
    }

    rm::Memory<int, rm::VRAM_CUDA> seq_gpu = seq;
    rm::Memory<int, rm::VRAM_CUDA> val_gpu(2);

    // compute sum and download
    
    sw();
    rm::sum(seq_gpu, val_gpu);
    el = sw();
    

    sw();
    rm::sum(seq_gpu, val_gpu);
    el = sw();

    // download
    rm::Memory<int, rm::RAM> val = val_gpu;


    std::cout << "Sum: " << val[0] << ", runtime " << el << " s" << std::endl;

    std::cout << val[1] << std::endl;

    std::cout << "GT:  " << (n_elements * n_elements + n_elements) / 2 << std::endl;
    // std::cout << "GT:  " << n_elements << std::endl;


    std::cout << "int max: " << std::numeric_limits<int>::max() << std::endl;
    std::cout << "i32 max: " << std::numeric_limits<int32_t>::max() << std::endl;
    // std::cout << "f32 max: " << std::numeric_limits<float>::max() << std::endl;



}


void preinit_cuda()
{
  int x = 1;
  rm::Memory<int, rm::VRAM_CUDA> bla = rm::MemoryView<int, rm::RAM>(&x, 1);

  rm::Memory<int, rm::RAM> y = rm::sum(bla);

  std::cout << y[0] << std::endl;
}

int main(int argc, char** argv)
{

  // preinit cuda
  preinit_cuda();
  

  test1();

  // test2();



  return 0;
}
