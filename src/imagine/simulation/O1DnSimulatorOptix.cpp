#include "imagine/simulation/O1DnSimulatorOptix.hpp"

#include "imagine/simulation/optix/OptixSimulationData.hpp"
#include "imagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

// Scan Programs
#include <imagine/simulation/optix/O1DnProgramRanges.hpp>
#include <imagine/simulation/optix/O1DnProgramNormals.hpp>
#include <imagine/simulation/optix/O1DnProgramGeneric.hpp>

namespace imagine
{

O1DnSimulatorOptix::O1DnSimulatorOptix(OptixMapPtr map)
:m_map(map)
,m_model(1)
,m_Tsb(1)
{
    m_programs.resize(2);
    
    m_programs[0].reset(new O1DnProgramRanges(map));
    m_programs[1].reset(new O1DnProgramNormals(map));

    CUDA_CHECK( cudaStreamCreate( &m_stream ) );

    Memory<Transform, RAM_CUDA> Tsb(1);
    Tsb->setIdentity();
    copy(Tsb, m_Tsb, m_stream);
}

O1DnSimulatorOptix::~O1DnSimulatorOptix()
{
    cudaStreamDestroy(m_stream);
}

void O1DnSimulatorOptix::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void O1DnSimulatorOptix::setModel(const O1DnModel<VRAM_CUDA>& model)
{
    Memory<O1DnModel<VRAM_CUDA>, RAM> mem(1);
    mem[0] = model;
    setModel(mem);
}

void O1DnSimulatorOptix::setModel(const O1DnModel<RAM>& model)
{
    Memory<O1DnModel<RAM>, RAM> mem(1);
    mem[0] = model;
    setModel(mem);
}

void O1DnSimulatorOptix::setModel(const Memory<O1DnModel<VRAM_CUDA>, RAM>& model)
{
    m_width = model->width;
    m_height = model->height;
    copy(model, m_model, m_stream);
}

void O1DnSimulatorOptix::setModel(const Memory<O1DnModel<RAM>, RAM>& model)
{
    Memory<O1DnModel<VRAM_CUDA>, RAM> model_tmp(1);
    // copy fields
    model_tmp->range = model->range;
    model_tmp->width = model->width;
    model_tmp->height = model->height;
    // upload to gpu
    model_tmp->rays = model->rays;
    setModel(model_tmp);
}

void O1DnSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<float, VRAM_CUDA>& ranges) const
{
    Memory<OptixSimulationDataRangesO1Dn, RAM> mem(1);
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->as.handle;
    mem->ranges = ranges.raw();

    Memory<OptixSimulationDataRangesO1Dn, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream);

    OptixProgramPtr program = m_programs[0];

    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream,
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataRangesO1Dn ),
                &program->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }
}

Memory<float, VRAM_CUDA> O1DnSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
    Memory<float, VRAM_CUDA> res(m_width * m_height * Tbm.size());
    simulateRanges(Tbm, res);
    return res;
}

void O1DnSimulatorOptix::simulateNormals(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<Vector, VRAM_CUDA>& normals) const
{
    Memory<OptixSimulationDataNormalsO1Dn, RAM> mem(1);
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->as.handle;
    mem->normals = normals.raw();

    Memory<OptixSimulationDataNormalsO1Dn, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream);

    OptixProgramPtr program = m_programs[1];

    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream,
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataNormalsO1Dn ),
                &program->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }
}

Memory<Vector, VRAM_CUDA> O1DnSimulatorOptix::simulateNormals(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
    Memory<Vector, VRAM_CUDA> res(m_width * m_height * Tbm.size());
    simulateNormals(Tbm, res);
    return res;
}

} // imagine