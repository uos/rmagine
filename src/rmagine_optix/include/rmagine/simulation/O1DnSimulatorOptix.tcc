#include <rmagine/util/optix/OptixDebug.hpp>
#include <rmagine/map/optix/OptixScene.hpp>
#include <rmagine/util/cuda/CudaStream.hpp>
#include <rmagine/simulation/optix/common.h>
#include <rmagine/simulation/optix/sim_pipelines.h>

namespace rmagine
{

template<typename BundleT>
void O1DnSimulatorOptix::preBuildProgram()
{
    if(!m_map)
    {
        throw std::runtime_error("[O1DnSimulatorOptix] preBuildProgram(): No Map available!");
    }

    OptixSimulationDataGeneric flags = OptixSimulationDataGeneric::Zero();
    flags.model_type = 2;
    set_generic_flags<BundleT>(flags);
    make_pipeline_sim(m_map->scene(), flags);
}

template<typename BundleT>
void O1DnSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    BundleT& res)
{
    if(!m_map)
    {
        throw std::runtime_error("[O1DnSimulatorOptix] simulate(): No Map available!");
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
    mem[0].model_type = 2;
    set_generic_flags(res, mem[0]);

    SimPipelinePtr program = make_pipeline_sim(m_map->scene(), mem[0]);

    // set general data
    // Memory<O1DnModel_<VRAM_CUDA>, VRAM_CUDA> model(1);
    // copy(m_model, model, m_stream->handle());

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
BundleT O1DnSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm)
{
    BundleT res;
    resize_memory_bundle<VRAM_CUDA>(res, m_width, m_height, Tbm.size());
    simulate(Tbm, res);
    return res;
}

} // namespace rmagine