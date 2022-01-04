#include "imagine/simulation/OnDnSimulatorOptix.hpp"

#include "imagine/simulation/optix/OptixSimulationData.hpp"
#include "imagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

// Scan Programs
#include <imagine/simulation/optix/OnDnProgramRanges.hpp>
#include <imagine/simulation/optix/OnDnProgramNormals.hpp>
#include <imagine/simulation/optix/OnDnProgramGeneric.hpp>

namespace imagine
{

OnDnSimulatorOptix::OnDnSimulatorOptix(OptixMapPtr map)
:m_map(map)
,m_model(1)
,m_Tsb(1)
{
    m_programs.resize(2);
    
    m_programs[0].reset(new OnDnProgramRanges(map));
    m_programs[1].reset(new OnDnProgramNormals(map));

    CUDA_CHECK( cudaStreamCreate( &m_stream ) );

    Memory<Transform, RAM_CUDA> Tsb(1);
    Tsb->setIdentity();
    copy(Tsb, m_Tsb, m_stream);
}

OnDnSimulatorOptix::~OnDnSimulatorOptix()
{
    cudaStreamDestroy(m_stream);
}

void OnDnSimulatorOptix::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void OnDnSimulatorOptix::setModel(const OnDnModel<VRAM_CUDA>& model)
{
    Memory<OnDnModel<VRAM_CUDA>, RAM> mem(1);
    mem[0] = model;
    setModel(mem);
}

void OnDnSimulatorOptix::setModel(const OnDnModel<RAM>& model)
{
    Memory<OnDnModel<RAM>, RAM> mem(1);
    mem[0] = model;
    setModel(mem);
}

void OnDnSimulatorOptix::setModel(const Memory<OnDnModel<VRAM_CUDA>, RAM>& model)
{
    m_width = model->width;
    m_height = model->height;
    copy(model, m_model, m_stream);
}

void OnDnSimulatorOptix::setModel(const Memory<OnDnModel<RAM>, RAM>& model)
{
    Memory<OnDnModel<VRAM_CUDA>, RAM> model_tmp(1);
    // copy fields
    model_tmp->range = model->range;
    model_tmp->width = model->width;
    model_tmp->height = model->height;
    // upload to gpu
    model_tmp->rays = model->rays;
    setModel(model_tmp);
}

void OnDnSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<float, VRAM_CUDA>& ranges) const
{
    Memory<OptixSimulationDataRangesOnDn, RAM> mem(1);
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->as.handle;
    mem->ranges = ranges.raw();

    Memory<OptixSimulationDataRangesOnDn, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream);

    OptixProgramPtr program = m_programs[0];

    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream,
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataRangesOnDn ),
                &program->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }
}

Memory<float, VRAM_CUDA> OnDnSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
    Memory<float, VRAM_CUDA> res(m_width * m_height * Tbm.size());
    simulateRanges(Tbm, res);
    return res;
}

void OnDnSimulatorOptix::simulateNormals(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<Vector, VRAM_CUDA>& normals) const
{
    Memory<OptixSimulationDataNormalsOnDn, RAM> mem(1);
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->as.handle;
    mem->normals = normals.raw();

    Memory<OptixSimulationDataNormalsOnDn, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream);

    OptixProgramPtr program = m_programs[1];

    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream,
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataNormalsOnDn ),
                &program->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }
}

Memory<Vector, VRAM_CUDA> OnDnSimulatorOptix::simulateNormals(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
    Memory<Vector, VRAM_CUDA> res(m_width * m_height * Tbm.size());
    simulateNormals(Tbm, res);
    return res;
}

} // imagine