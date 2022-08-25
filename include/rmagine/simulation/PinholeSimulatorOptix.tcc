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

    OptixSimulationDataGeneric flags;
    flags.model_type = 1;
    setGenericFlags<BundleT>(flags);
    make_pipeline_sim(m_map->scene(), flags);
}

template<typename BundleT>
void PinholeSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    BundleT& res)
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
    mem[0].model_type = 1;
    setGenericFlags(res, mem[0]);

    SimPipelinePtr program = make_pipeline_sim(m_map->scene(), mem[0]);

    // set general data
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->scene()->as()->handle;

    // set generic data
    setGenericData(res, mem[0]);

    // 10000 velodynes 
    // - upload Params: 0.000602865s
    // - launch: 5.9642e-05s
    // => this takes too long. Can we somehow preupload stuff?
    Memory<OptixSimulationDataGeneric, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream->handle());

    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream->handle(),
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataGeneric ),
                program->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }

}

template<typename BundleT>
BundleT PinholeSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm)
{
    BundleT res;
    resizeMemoryBundle<VRAM_CUDA>(res, m_width, m_height, Tbm.size());
    simulate(Tbm, res);
    return res;
}

} // namespace rmagine