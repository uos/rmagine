#include "rmagine/simulation/SphereSimulatorOptix.hpp"

#include "rmagine/simulation/optix/sim_program_data.h"
#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

// Scan Programs
// #include <rmagine/simulation/optix/SphereProgramRanges.hpp>
// #include <rmagine/simulation/optix/SphereProgramGeneric.hpp>

#include <rmagine/util/cuda/CudaStream.hpp>

namespace rmagine
{

SphereSimulatorOptix::SphereSimulatorOptix()
:SimulatorOptix()
,m_model(1)
{
  
}

SphereSimulatorOptix::SphereSimulatorOptix(OptixMapPtr map)
:SimulatorOptix(map)
,m_model(1)
{
  
}

// SphereSimulatorOptix::SphereSimulatorOptix(OptixScenePtr scene)
// :SphereSimulatorOptix()
// {
//   OptixMapPtr map = std::make_shared<OptixMap>(scene);
//   setMap(map);
// }

SphereSimulatorOptix::~SphereSimulatorOptix()
{
  
}
    
void SphereSimulatorOptix::setModel(const Memory<SphericalModel, RAM>& model)
{
  m_width = model->getWidth();
  m_height = model->getHeight();
  m_model = model;

  Memory<SensorModelUnion, RAM> model_union(1);
  model_union->spherical = m_model.raw();
  m_model_union = model_union;
}

void SphereSimulatorOptix::setModel(const SphericalModel& model)
{
  Memory<SphericalModel, RAM> tmp(1);
  tmp[0] = model;
  setModel(tmp);
}

void SphereSimulatorOptix::launch(
    const Memory<OptixSimulationDataGeneric, RAM>& mem,
    const PipelinePtr program) const
{
  Memory<OptixSimulationDataGeneric, VRAM_CUDA> d_mem(1);
  copy(mem, d_mem, m_stream);

  RM_OPTIX_CHECK( optixLaunch(
              program->pipeline,
              m_stream->handle(),
              reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
              sizeof( OptixSimulationDataGeneric ),
              program->sbt,
              m_width, // width Xdim
              m_height, // height Ydim
              mem->Nposes // depth Zdim
              ) );
}

} // rmagine