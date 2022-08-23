
// #include <rmagine/simulation/optix/SphereProgramGeneric.hpp>
#include <rmagine/simulation/optix/SensorProgramGeneric.hpp>
#include <rmagine/util/optix/OptixDebug.hpp>
#include <optix.h>
#include <optix_stubs.h>

// #include <rmagine/util/StopWatch.hpp>
#include <rmagine/map/optix/OptixScene.hpp>
#include <rmagine/util/cuda/CudaStream.hpp>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/simulation/optix/common.h>

namespace rmagine
{


template<typename BundleT>
void SphereSimulatorOptix::preBuildProgram()
{
    if(!m_map)
    {
        throw std::runtime_error("[SphereSimulatorOptix] preBuildProgram(): No Map available!");
    }

    OptixSimulationDataGeneric flags;
    flags.model_type = 0;
    setGenericFlags<BundleT>(flags);
    m_map->scene()->registerSensorProgram(flags);
}

template<typename BundleT>
void SphereSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    BundleT& res)
{
    // StopWatch sw;
    // double el;

    if(!m_map)
    {
        // no map set
        throw std::runtime_error("[SphereSimulatorOptix] simulate(): No Map available!");
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
    mem[0].model_type = 0;
    setGenericFlags(res, mem[0]);

    OptixSensorProgram program = m_map->scene()->registerSensorProgram(mem[0]);

    // set general data
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model_union.raw();
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
    
    if(program.pipeline)
    {
        OPTIX_CHECK( optixLaunch(
                program.pipeline->pipeline,
                m_stream->handle(),
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataGeneric ),
                &program.sbt->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }
}

template<typename BundleT>
BundleT SphereSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm)
{
    BundleT res;
    resizeMemoryBundle<VRAM_CUDA>(res, m_width, m_height, Tbm.size());
    simulate(Tbm, res);
    return res;
}

} // namespace rmagine