#include <rmagine/simulation/optix/OnDnProgramGeneric.hpp>
#include <rmagine/util/optix/OptixDebug.hpp>
#include <optix.h>
#include <optix_stubs.h>

// #include <rmagine/util/StopWatch.hpp>
#include <rmagine/simulation/optix/common.h>

namespace rmagine
{

template<typename BundleT>
void OnDnSimulatorOptix::preBuildProgram()
{
    if(!m_map)
    {
        throw std::runtime_error("[OnDnSimulatorOptix] preBuildProgram(): No Map available!");
    }

    OptixSimulationDataGenericOnDn flags;
    setGenericFlags<BundleT>(flags);
    auto it = m_generic_programs.find(flags);
    
    if(it == m_generic_programs.end())
    {
        m_generic_programs[flags] = std::make_shared<OnDnProgramGeneric>(m_map, flags);
    }
}

template<typename BundleT>
void OnDnSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    BundleT& res)
{
    if(!m_map)
    {
        // no map set
        throw std::runtime_error("[OnDnSimulatorOptix] simulate(): No Map available!");
        return;
    }

    auto optix_ctx = m_map->context();
    auto cuda_ctx = optix_ctx->getCudaContext();
    if(!cuda_ctx->isActive())
    {
        std::cout << "[OnDnSimulatorOptix::simulate() Need to activate map context" << std::endl;
        cuda_ctx->use();
    }


    Memory<OptixSimulationDataGenericOnDn, RAM> mem(1);
    setGenericFlags(res, mem[0]);

    auto it = m_generic_programs.find(mem[0]);
    OptixProgramPtr program;
    if(it == m_generic_programs.end())
    {
        program.reset(new OnDnProgramGeneric(m_map, mem[0] ) );
        m_generic_programs[mem[0]] = program;
    } else {
        program = it->second;
        program->updateSBT();
    }

    // set general data

    Memory<OnDnModel_<VRAM_CUDA>, VRAM_CUDA> model(1);
    copy(m_model, model, m_stream);

    mem->Tsb = m_Tsb.raw();
    mem->model = model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->scene()->as()->handle;

    // set generic data
    setGenericData(res, mem[0]);

    // 10000 velodynes 
    // - upload Params: 0.000602865s
    // - launch: 5.9642e-05s
    // => this takes too long. Can we somehow preupload stuff?
    Memory<OptixSimulationDataGenericOnDn, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream->handle());

    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream->handle(),
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataGenericOnDn ),
                &program->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }

}

template<typename BundleT>
BundleT OnDnSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm)
{
    BundleT res;
    resizeMemoryBundle<VRAM_CUDA>(res, m_width, m_height, Tbm.size());
    simulate(Tbm, res);
    return res;
}

} // namespace rmagine