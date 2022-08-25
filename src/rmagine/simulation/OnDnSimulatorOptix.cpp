#include "rmagine/simulation/OnDnSimulatorOptix.hpp"

#include "rmagine/simulation/optix/OptixSimulationData.hpp"
#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

// Scan Programs
// #include <rmagine/simulation/optix/OnDnProgramRanges.hpp>

#include <rmagine/util/Debug.hpp>

#include <rmagine/util/cuda/CudaStream.hpp>

namespace rmagine
{


OnDnSimulatorOptix::OnDnSimulatorOptix()
:m_model(1)
,m_Tsb(1)
{
    Memory<Transform, RAM_CUDA> I(1);
    I->setIdentity();
    m_Tsb = I;
}

OnDnSimulatorOptix::OnDnSimulatorOptix(OptixMapPtr map)
:OnDnSimulatorOptix()
{
    setMap(map);
}

OnDnSimulatorOptix::~OnDnSimulatorOptix()
{
    // m_programs.resize(0);
}

void OnDnSimulatorOptix::setMap(OptixMapPtr map)
{
    m_map = map;

    // m_programs.resize(2);
    // m_programs[0].reset(new OnDnProgramRanges(map));
    // m_programs[1].reset(new OnDnProgramNormals(map));

    // need to create stream after map was created: cuda device api context is required
    m_stream = m_map->stream();
}

void OnDnSimulatorOptix::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void OnDnSimulatorOptix::setTsb(const Transform& Tsb)
{
    Memory<Transform, RAM> tmp(1);
    tmp[0] = Tsb;
    setTsb(tmp);
}

void OnDnSimulatorOptix::setModel(const OnDnModel_<VRAM_CUDA>& model)
{
    m_width = model.getWidth();
    m_height = model.getHeight();

    m_model.resize(1);
    m_model[0] = model;

    m_model_d.resize(1);
    copy(m_model, m_model_d, m_stream->handle());

    Memory<SensorModelUnion, RAM> model_union(1);
    model_union->ondn = m_model_d.raw();
    m_model_union = model_union;
}

void OnDnSimulatorOptix::setModel(const OnDnModel_<RAM>& model)
{
    OnDnModel_<VRAM_CUDA> model_gpu;
    model_gpu.width = model.width;
    model_gpu.height = model.height;
    model_gpu.range = model.range;

    // upload ray data
    model_gpu.dirs = model.dirs;
    model_gpu.origs = model.origs;

    setModel(model_gpu);
}

void OnDnSimulatorOptix::setModel(const Memory<OnDnModel_<VRAM_CUDA>, RAM>& model)
{
    m_width = model->width;
    m_height = model->height;
    
    setModel(model[0]);

    TODO_TEST_FUNCTION
}

void OnDnSimulatorOptix::setModel(const Memory<OnDnModel_<RAM>, RAM>& model)
{
    // TODO: test
    TODO_NOT_IMPLEMENTED
}

void OnDnSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<float, VRAM_CUDA>& ranges) const
{
    TODO_NOT_IMPLEMENTED

    // TODO: how to do this before?
    // Memory<OnDnModel_<VRAM_CUDA>, VRAM_CUDA> model(1);
    // copy(m_model, model, m_stream);

    // Memory<OptixSimulationDataRangesOnDn, RAM> mem(1);
    // mem->Tsb = m_Tsb.raw();
    // mem->model = model.raw();
    // mem->Tbm = Tbm.raw();
    // mem->handle = m_map->scene()->as()->handle;
    // mem->ranges = ranges.raw();

    // Memory<OptixSimulationDataRangesOnDn, VRAM_CUDA> d_mem(1);
    // copy(mem, d_mem, m_stream);

    // OptixProgramPtr program = m_programs[0];

    // if(program)
    // {
    //     OPTIX_CHECK( optixLaunch(
    //             program->pipeline,
    //             m_stream,
    //             reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
    //             sizeof( OptixSimulationDataRangesOnDn ),
    //             &program->sbt,
    //             m_width, // width Xdim
    //             m_height, // height Ydim
    //             Tbm.size() // depth Zdim
    //             ));
    // } else {
    //     throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    // }
}

Memory<float, VRAM_CUDA> OnDnSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
    Memory<float, VRAM_CUDA> res(m_width * m_height * Tbm.size());
    simulateRanges(Tbm, res);
    return res;
}

void OnDnSimulatorOptix::launch(
    const Memory<OptixSimulationDataGeneric, RAM>& mem,
    PipelinePtr program)
{
    Memory<OptixSimulationDataGeneric, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream->handle());

    OPTIX_CHECK( optixLaunch(
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