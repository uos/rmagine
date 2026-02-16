#include <iostream>

// CPU - Embree
#include <rmagine/simulation/SphereSimulatorEmbree.hpp>
#include <rmagine/simulation/PinholeSimulatorEmbree.hpp>

#include "benchmarks/velodyne_benchmark.hpp"

using namespace rmagine;

int main(int argc, char** argv)
{
  std::cout << "Rmagine Benchmark Embree" << std::endl;

  // minimum 1 arguments
  if(argc < 2)
  {
      std::cout << "Usage: " << argv[0] << " mesh_file" << std::endl;
      return 0;
  }

  const std::string path_to_mesh = argv[1];    
  std::cout << "Mesh: " << path_to_mesh << std::endl;

  // load scene and create simulator
  EmbreeMapPtr cpu_mesh = import_embree_map(path_to_mesh);
  auto cpu_sim = std::make_shared<SphereSimulatorEmbree>(cpu_mesh);

  VelodyneBenchmarkConfig config {
        // Total runtime of the Benchmark in seconds
        .duration = 10.0,
        // Poses to check per call
        .n_poses = 5 * 1024
      };

  velodyne_benchmark(cpu_sim, config);

  return 0;
}
