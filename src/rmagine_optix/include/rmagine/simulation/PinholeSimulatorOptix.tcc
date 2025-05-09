#include <rmagine/simulation/optix/sim_pipelines.h>
#include <rmagine/util/optix/OptixDebug.hpp>

// #include <rmagine/util/StopWatch.hpp>
#include <rmagine/simulation/optix/common.h>


namespace rmagine
{

template<typename BundleT>
void PinholeSimulatorOptix::preBuildProgram()
{
  if(!m_map)
  {
    throw std::runtime_error("[PinholeSimulatorOptix] preBuildProgram(): No Map available!");
  }

  OptixSimulationDataGeneric flags = OptixSimulationDataGeneric::Zero();
  flags.model_type = 1;
  set_generic_flags<BundleT>(flags);
  make_pipeline_sim(m_map->scene(), flags);
}

template<typename BundleT>
void PinholeSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    BundleT& res) const
{
  if(!m_map)
  {
    // no map set
    throw std::runtime_error("[PinholeSimulatorOptix] simulate(): No Map available!");
    return;
  }

  auto optix_ctx = m_map->context();
  auto cuda_ctx = optix_ctx->getCudaContext();
  if(!cuda_ctx->isActive())
  {
    std::cout << "[PinholeSimulatorOptix::simulate() Need to activate map context" << std::endl;
    cuda_ctx->use();
  }

  Memory<OptixSimulationDataGeneric, RAM> mem(1);
  mem[0] = OptixSimulationDataGeneric::Zero();
  mem[0].model_type = 1;
  set_generic_flags(res, mem[0]);

  PipelinePtr program = make_pipeline_sim(m_map->scene(), mem[0]);

  // set general data
  mem->Tsb = m_Tsb.raw();
  mem->model = m_model_union.raw();
  mem->Tbm = Tbm.raw();
  mem->Nposes = Tbm.size();
  mem->handle = m_map->scene()->as()->handle;

  // set generic data
  set_generic_data(res, mem[0]);

  launch(mem, program);
}

template<typename BundleT>
BundleT PinholeSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
  BundleT res;
  resize_memory_bundle<VRAM_CUDA>(res, m_width, m_height, Tbm.size());
  simulate(Tbm, res);
  return res;
}

} // namespace rmagine