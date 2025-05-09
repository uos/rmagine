#include "rmagine/simulation/PinholeSimulatorOptix.hpp"

#include "rmagine/simulation/optix/sim_program_data.h"
#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

// Scan Programs
// #include <rmagine/simulation/optix/PinholeProgramRanges.hpp>
// #include <rmagine/simulation/optix/PinholeProgramNormals.hpp>

namespace rmagine
{

PinholeSimulatorOptix::PinholeSimulatorOptix()
:SimulatorOptix()
,m_model(1)
{
  
}

PinholeSimulatorOptix::PinholeSimulatorOptix(OptixMapPtr map)
:SimulatorOptix(map)
,m_model(1)
{
  
}

PinholeSimulatorOptix::~PinholeSimulatorOptix()
{
  
}

void PinholeSimulatorOptix::setModel(const Memory<PinholeModel, RAM>& model)
{
  m_width = model->width;
  m_height = model->height;
  m_model = model;

  Memory<SensorModelUnion, RAM> model_union(1);
  model_union->pinhole = m_model.raw();
  m_model_union = model_union;
}

void PinholeSimulatorOptix::setModel(const PinholeModel& model)
{
  Memory<PinholeModel, RAM> tmp(1);
  tmp[0] = model;
  setModel(tmp);
}

void PinholeSimulatorOptix::launch(
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