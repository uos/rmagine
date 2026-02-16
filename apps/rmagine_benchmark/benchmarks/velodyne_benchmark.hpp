#ifndef RMAGINE_BENCHMARK_VELODYNE_BENCHMARK_HPP
#define RMAGINE_BENCHMARK_VELODYNE_BENCHMARK_HPP

#include <iostream>
#include <string>
#include <memory>

// Core rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>

namespace rmagine
{

Memory<LiDARModel, RAM> velodyne_model()
{
    Memory<LiDARModel, RAM> model(1);
    model->theta.min = -M_PI;
    model->theta.inc = 0.4 * M_PI / 180.0;
    model->theta.size = 900;

    model->phi.min = -15.0 * M_PI / 180.0;
    model->phi.inc = 2.0 * M_PI / 180.0;
    model->phi.size = 16;
    
    model->range.min = 0.1;
    model->range.max = 130.0;
    return model;
}

struct VelodyneBenchmarkConfig
{
  double duration;
  size_t n_poses;
};

/**
 * @brief Benchmark for Velodyne-like LiDAR simulation. Measures the number of Velodyne scans per second.
 * A Velodyne scan is defined as the number of rays in the LiDAR model. For example, a Velodyne VLP-16 has 16 vertical channels and 900 horizontal channels, resulting in 16*900 = 14,400 rays per scan.
 * The benchmarks input are a config and the specification of input and output types
 * The input/output memory type influences the the total runtime in that sense that conversions may has to take place.
 */
template<typename InputMemType, typename OutputMemType, typename SimT>
void velodyne_benchmark(
  std::shared_ptr<SimT> sim,
  const VelodyneBenchmarkConfig& config)
{ 
  std::cout << "Inputs: " << std::endl;
  std::cout << "- nposes: " << config.n_poses << std::endl;
  std::cout << "- duration: " << config.duration << " seconds" << std::endl;

  StopWatch sw;
  double elapsed;
  double elapsed_total = 0.0;

  // Define one Transform Sensor to Base
  Transform Tsb = {
    .R = {0.0, 0.0, 0.0, 1.0},
    .t = {0.0, 0.0, 0.0},
    .stamp = 0
  };
  sim->setTsb(Tsb);

  Memory<LiDARModel, RAM> model = velodyne_model();
  sim->setModel(model);
  std::cout << "Unit: 1 Velodyne scan (velo) = " << model[0].size() << " Rays" << std::endl;

  // Define Transforms Base to Map (Poses)
  Memory<Transform, RAM> Tbm(config.n_poses);
  for(size_t i=0; i<Tbm.size(); i++)
  {
    Tbm[i] = Tsb;
  }

  // upload poses to device if necessary
  Memory<Transform, InputMemType> Tbm_device = Tbm;

  double velos_per_second_mean = 0.0;

  std::cout << "-- Starting Benchmark --" << std::endl;

  using ResultT = Bundle<
    Ranges<OutputMemType>
  >;

  ResultT res;
  res.ranges.resize(Tbm.size() * model->phi.size * model->theta.size);

  int run = 0;
  while(elapsed_total < config.duration)
  {
    double n_dbl = static_cast<double>(run) + 1.0;
    // Simulate
    sw();
    sim->simulate(Tbm_device, res);
    elapsed = sw();
    elapsed_total += elapsed;
    const double velos_per_second = static_cast<double>(config.n_poses) / elapsed;
    velos_per_second_mean = (n_dbl - 1.0)/(n_dbl) * velos_per_second_mean + (1.0 / n_dbl) * velos_per_second; 
    
    std::cout 
    << std::fixed
    << "[ " << int((elapsed_total / config.duration)*100.0) << "%" << " - " 
    << velos_per_second << " velos/s" 
    << ", mean: " << velos_per_second_mean << " velos/s] \r";
    std::cout.flush();

    run++;
  }

  std::cout << std::endl;
  std::cout << "Result: " << velos_per_second_mean << " velos/s" << std::endl;
}

} // namespace rmagine

#endif // RMAGINE_BENCHMARK_VELODYNE_BENCHMARK_HPP