#include "rmagine/simulation/O1DnSimulatorOptix.hpp"

#include "rmagine/simulation/optix/OptixSimulationData.hpp"
#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

// Scan Programs
#include <rmagine/simulation/optix/O1DnProgramRanges.hpp>
#include <rmagine/simulation/optix/O1DnProgramNormals.hpp>
#include <rmagine/simulation/optix/O1DnProgramGeneric.hpp>

#include <rmagine/util/Debug.hpp>

namespace rmagine
{

O1DnSimulatorOptix::O1DnSimulatorOptix()
:m_Tsb(1)
{

}

O1DnSimulatorOptix::O1DnSimulatorOptix(OptixMapPtr map)
:O1DnSimulatorOptix()
{
    setMap(map);

    Memory<Transform, RAM_CUDA> Tsb(1);
    Tsb->setIdentity();
    copy(Tsb, m_Tsb, m_stream);
}

O1DnSimulatorOptix::~O1DnSimulatorOptix()
{
    m_programs.clear();
    m_generic_programs.clear();
    cudaStreamDestroy(m_stream);
}

void O1DnSimulatorOptix::setMap(OptixMapPtr map)
{
    m_map = map;

    m_programs.resize(2);
    
    m_programs[0].reset(new O1DnProgramRanges(map));
    m_programs[1].reset(new O1DnProgramNormals(map));

    CUDA_CHECK( cudaStreamCreate( &m_stream ) );
}

void O1DnSimulatorOptix::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void O1DnSimulatorOptix::setTsb(const Transform& Tsb)
{
    Memory<Transform, RAM> tmp(1);
    tmp[0] = Tsb;
    setTsb(tmp);
}

void O1DnSimulatorOptix::setModel(const O1DnModel_<VRAM_CUDA>& model)
{
    m_width = model.getWidth();
    m_height = model.getHeight();

    m_model.resize(1);
    m_model[0] = model;
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
    TODO_NOT_IMPLEMENTED
}

void O1DnSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<float, VRAM_CUDA>& ranges) const
{
    if(!m_map)
    {
        throw std::runtime_error("[O1DnSimulatorOptix] simulateRanges(): No Map available!");
    }

    Memory<O1DnModel_<VRAM_CUDA>, VRAM_CUDA> model(1);
    copy(m_model, model, m_stream);

    Memory<OptixSimulationDataRangesO1Dn, RAM> mem(1);
    mem->Tsb = m_Tsb.raw();
    mem->model = model.raw();
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
    Memory<O1DnModel_<VRAM_CUDA>, VRAM_CUDA> model(1);
    copy(m_model, model, m_stream);

    Memory<OptixSimulationDataNormalsO1Dn, RAM> mem(1);
    mem->Tsb = m_Tsb.raw();
    mem->model = model.raw();
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

} // rmagine