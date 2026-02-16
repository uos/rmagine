#ifndef RMAGINE_OPTIX_SIMULATION_SIMULATOR_OPTIX_HPP
#define RMAGINE_OPTIX_SIMULATION_SIMULATOR_OPTIX_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/optix_modules.h>

#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/sensor_models.h>

// Generic
#include <rmagine/simulation/SimulationResults.hpp>
#include <rmagine/types/Bundle.hpp>
#include <rmagine/simulation/optix/sim_program_data.h>

#include <cuda_runtime.h>

#include <rmagine/util/cuda/cuda_definitions.h>

namespace rmagine
{

class SimulatorOptix
{
public:
  SimulatorOptix();
  SimulatorOptix(OptixMapPtr map);

  virtual ~SimulatorOptix();

  void setMap(OptixMapPtr map);

  void setTsb(const Memory<Transform, RAM>& Tsb);
  void setTsb(const Transform& Tsb);

  inline OptixMapPtr map() const 
  {
    return m_map;
  }

  /**
   * @brief Simulate from one pose
   * 
   * @tparam BundleT 
   * @param Tbm Transform from base to map. aka pose in map
   */
  template<typename BundleT>
  void simulate(const Transform& Tbm, BundleT& ret) const;

  template<typename BundleT>
  BundleT simulate(const Transform& Tbm) const;

  /**
   * @brief Simulation of a LiDAR-Sensor in a given mesh
   * 
   * @tparam ResultT Pass desired results via ResultT=Bundle<...>;
   * @param Tbm Transformations between base and map. eg Poses or Particles. In VRAM
   * @return ResultT
   */
  template<typename BundleT>
  BundleT simulate(
      const Memory<Transform, VRAM_CUDA>& Tbm) const;

  // template<typename BundleT>
  // virtual void simulate(
  //     const Memory<Transform, VRAM_CUDA>& Tbm,
  //     BundleT& res) const = 0;

protected:

  OptixMapPtr m_map;
  CudaStreamPtr m_stream;
  
  Memory<Transform, VRAM_CUDA> m_Tsb;

  // generic model parameter
  uint32_t m_width;
  uint32_t m_height;
  Memory<SensorModelUnion, VRAM_CUDA> m_model_union;
};

} // namespace rmagine

#include "SimulatorOptix.tcc"

#endif // RMAGINE_OPTIX_SIMULATION_SIMULATOR_OPTIX_HPP