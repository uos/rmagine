#include <iostream>

// GPU - Optix
#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/types/MemoryCuda.hpp>

#include "benchmarks/velodyne_benchmark.hpp"

using namespace rmagine;

int main(int argc, char** argv)
{
  std::cout << "Rmagine Benchmark OptiX" << std::endl;

  // minimum 1 argument
  if(argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " mesh_file" << std::endl;
    return 0;
  }

  const std::string path_to_mesh = argv[1];
  std::cout << "Mesh: " << path_to_mesh << std::endl;

  // load scene and create simulator
  OptixMapPtr gpu_mesh = import_optix_map(path_to_mesh);
  SphereSimulatorOptixPtr gpu_sim = std::make_shared<SphereSimulatorOptix>(gpu_mesh);

  VelodyneBenchmarkConfig config {
        // Total runtime of the Benchmark in seconds
        .duration = 10.0,
        // Poses to check per call
        .n_poses = 10 * 1024
      };

  velodyne_benchmark(gpu_sim, config);

  return 0;
}