#include <iostream>

// GPU - Vulkan
#include <rmagine/simulation/SphereSimulatorVulkan.hpp>
#include <rmagine/map/VulkanMap.hpp>
#include <rmagine/types/MemoryVulkan.hpp>

#include "benchmarks/velodyne_benchmark.hpp"

using namespace rmagine;

int main(int argc, char** argv)
{
  std::cout << "Rmagine Benchmark Vulkan" << std::endl;

  // minimum 1 argument
  if(argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " mesh_file" << std::endl;
    return 0;
  }

  const std::string path_to_mesh = argv[1];
  std::cout << "Mesh: " << path_to_mesh << std::endl;

  // load scene and create simulator
  VulkanMapPtr gpu_mesh = import_vulkan_map(path_to_mesh);
  SphereSimulatorVulkanPtr gpu_sim = std::make_shared<SphereSimulatorVulkan>(gpu_mesh);

  VelodyneBenchmarkConfig config {
        // Total runtime of the Benchmark in seconds
        .duration = 10.0,
        // Poses to check per call
        .n_poses = 10 * 1024
      };

  // template types: InputMemType, OutputMemType
  velodyne_benchmark<DEVICE_LOCAL_VULKAN, DEVICE_LOCAL_VULKAN>(gpu_sim, config);

  return 0;
}