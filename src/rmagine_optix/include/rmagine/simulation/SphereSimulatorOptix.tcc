#include <rmagine/util/optix/OptixDebug.hpp>
#include <rmagine/map/optix/OptixScene.hpp>
#include <rmagine/util/cuda/CudaStream.hpp>
#include <rmagine/simulation/optix/common.h>
#include <rmagine/simulation/optix/sim_pipelines.h>

namespace rmagine
{


template<typename BundleT>
void SphereSimulatorOptix::preBuildProgram()
{
  if(!m_map)
  {
    throw std::runtime_error("[SphereSimulatorOptix] preBuildProgram(): No Map available!");
  }

  OptixSimulationDataGeneric flags = OptixSimulationDataGeneric::Zero();
  flags.model_type = 0;
  set_generic_flags<BundleT>(flags);
  make_pipeline_sim(m_map->scene(), flags);
}

template<typename BundleT>
void SphereSimulatorOptix::simulate(const Transform& Tbm, BundleT& ret) const
{
  // upload pose
  Transform Tbm_tmp = Tbm;
  const MemoryView<Transform, RAM> Tbm_mem(&Tbm_tmp, 1);
  Memory<Transform, VRAM_CUDA> Tbm_gpu = Tbm_mem;
  simulate(Tbm_gpu, ret);
}

template<typename BundleT>
BundleT SphereSimulatorOptix::simulate(const Transform& Tbm) const
{
  BundleT res;
  resize_memory_bundle<VRAM_CUDA>(res, m_model->getWidth(), m_model->getHeight(), 1);
  simulate(Tbm, res);
  return res;
}

template<typename BundleT>
void SphereSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    BundleT& res) const
{
  if(!m_map)
  {
    // no map set
    throw std::runtime_error("[SphereSimulatorOptix] simulate(): No Map available!");
    return;
  }

  if(m_map->scene()->type() == OptixSceneType::NONE)
  {
    // SCENE EMPTY
    // TODO: fill values with invalid values
    return;
  }

  auto optix_ctx = m_map->context();
  auto cuda_ctx = optix_ctx->getCudaContext();
  if(!cuda_ctx->isActive())
  {
    std::cout << "[SphereSimulatorOptix::simulate() Need to activate map context" << std::endl;
    cuda_ctx->use();
  }

  Memory<OptixSimulationDataGeneric, RAM> mem(1);
  mem[0] = OptixSimulationDataGeneric::Zero();
  mem[0].model_type = 0;
  set_generic_flags(res, mem[0]);

  SimPipelinePtr program = make_pipeline_sim(m_map->scene(), mem[0]);

  // set general data
  mem->Tsb = m_Tsb.raw();
  mem->model = m_model_union.raw();
  mem->Tbm = Tbm.raw();
  mem->Nposes = Tbm.size();
  mem->handle = m_map->scene()->as()->handle;
  
  // set generic data
  set_generic_data(res, mem[0]);

  // launch
  launch(mem, program);
}

template<typename BundleT>
BundleT SphereSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
  BundleT res;
  resize_memory_bundle<VRAM_CUDA>(res, m_width, m_height, Tbm.size());
  simulate(Tbm, res);
  return res;
}

} // namespace rmagine