#include "rmagine/simulation/O1DnSimulatorOptix.hpp"

#include "rmagine/simulation/optix/sim_program_data.h"
#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

// Scan Programs
// #include <rmagine/simulation/optix/O1DnProgramRanges.hpp>
// #include <rmagine/simulation/optix/O1DnProgramNormals.hpp>

#include <rmagine/util/Debug.hpp>

#include <rmagine/util/cuda/CudaStream.hpp>

namespace rmagine
{

O1DnSimulatorOptix::O1DnSimulatorOptix()
:SimulatorOptix()
,m_model(1)
{
  
}

O1DnSimulatorOptix::O1DnSimulatorOptix(OptixMapPtr map)
:SimulatorOptix(map)
,m_model(1)
{
  
}

O1DnSimulatorOptix::~O1DnSimulatorOptix()
{

}

void O1DnSimulatorOptix::setModel(const O1DnModel_<VRAM_CUDA>& model)
{
    m_width = model.getWidth();
    m_height = model.getHeight();

    m_model.resize(1);
    m_model[0] = model;

    m_model_d.resize(1);
    copy(m_model, m_model_d, m_stream);

    Memory<SensorModelUnion, RAM> model_union(1);
    model_union->o1dn = m_model_d.raw();
    m_model_union = model_union;
}

void O1DnSimulatorOptix::setModel(const O1DnModel_<RAM>& model)
{
  O1DnModel_<VRAM_CUDA> model_gpu;
  model_gpu.width = model.width;
  model_gpu.height = model.height;
  model_gpu.range = model.range;

  // upload ray data
  model_gpu.dirs = model.dirs;
  model_gpu.orig = model.orig;

  setModel(model_gpu);
}

void O1DnSimulatorOptix::setModel(const Memory<O1DnModel_<VRAM_CUDA>, RAM>& model)
{
  m_width = model->width;
  m_height = model->height;
  setModel(model[0]);
  TODO_TEST_FUNCTION
}

void O1DnSimulatorOptix::setModel(const Memory<O1DnModel_<RAM>, RAM>& model)
{
  setModel(model[0]);
  TODO_TEST_FUNCTION
}

void O1DnSimulatorOptix::launch(
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