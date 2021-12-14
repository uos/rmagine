#include "imagine/simulation/SphereSimulatorOptix.hpp"

#include "imagine/simulation/optix/OptixSimulationData.hpp"
#include "imagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

// Scan Programs
#include <imagine/simulation/optix/SphereProgramRanges.hpp>
#include <imagine/simulation/optix/SphereProgramNormals.hpp>
#include <imagine/simulation/optix/SphereProgramGeneric.hpp>

namespace imagine
{

SphereSimulatorOptix::SphereSimulatorOptix(OptixMapPtr map)
:m_map(map)
{
    m_programs.resize(2);
    
    // programs[0].reset(new ScanProgramHit(mesh));
    m_programs[0].reset(new SphereProgramRanges(map));
    m_programs[1].reset(new SphereProgramNormals(map));

    CUDA_CHECK( cudaStreamCreate( &m_stream ) );
}

SphereSimulatorOptix::~SphereSimulatorOptix()
{
    cudaStreamDestroy(m_stream);
}

void SphereSimulatorOptix::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}
    
void SphereSimulatorOptix::setModel(const Memory<LiDARModel, RAM>& model)
{
    m_width = model->theta.size;
    m_height = model->phi.size;
    m_model = model;
}

void SphereSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<float, VRAM_CUDA>& ranges) const
{
    Memory<OptixSimulationDataRangesSphere, RAM> mem;
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->as.handle;
    mem->ranges = ranges.raw();

    Memory<OptixSimulationDataRangesSphere, VRAM_CUDA> d_mem;
    copy(mem, d_mem, m_stream);

    OptixProgramPtr program = m_programs[0];

    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream,
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataRangesSphere ),
                &program->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }
}

Memory<float, VRAM_CUDA> SphereSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
    Memory<float, VRAM_CUDA> res(m_width * m_height * Tbm.size());
    simulateRanges(Tbm, res);
    return res;
}

void SphereSimulatorOptix::simulateNormals(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<Vector, VRAM_CUDA>& normals) const
{
    Memory<OptixSimulationDataNormalsSphere, RAM> mem;
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->as.handle;
    mem->normals = normals.raw();

    Memory<OptixSimulationDataNormalsSphere, VRAM_CUDA> d_mem;
    copy(mem, d_mem, m_stream);

    OptixProgramPtr program = m_programs[1];

    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream,
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataNormalsSphere ),
                &program->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }
}

Memory<Vector, VRAM_CUDA> SphereSimulatorOptix::simulateNormals(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
    Memory<Vector, VRAM_CUDA> res(m_width * m_height * Tbm.size());
    simulateNormals(Tbm, res);
    return res;
}



} // imagine