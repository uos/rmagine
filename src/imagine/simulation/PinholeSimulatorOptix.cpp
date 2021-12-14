#include "imagine/simulation/PinholeSimulatorOptix.hpp"

#include "imagine/simulation/optix/OptixSimulationData.hpp"
#include "imagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

// Scan Programs
#include <imagine/simulation/optix/PinholeProgramRanges.hpp>
#include <imagine/simulation/optix/PinholeProgramNormals.hpp>
#include <imagine/simulation/optix/PinholeProgramGeneric.hpp>

namespace imagine
{

PinholeSimulatorOptix::PinholeSimulatorOptix(OptixMapPtr map)
:m_map(map)
{
    m_programs.resize(2);
    
    // programs[0].reset(new ScanProgramHit(mesh));
    m_programs[0].reset(new PinholeProgramRanges(map));
    m_programs[1].reset(new PinholeProgramNormals(map));

    CUDA_CHECK( cudaStreamCreate( &m_stream ) );
}

PinholeSimulatorOptix::~PinholeSimulatorOptix()
{
    cudaStreamDestroy(m_stream);
}

void PinholeSimulatorOptix::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}
    
void PinholeSimulatorOptix::setModel(const Memory<PinholeModel, RAM>& model)
{
    m_width = model->width;
    m_height = model->height;
    m_model = model;
}

void PinholeSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<float, VRAM_CUDA>& ranges) const
{
    Memory<OptixSimulationDataRangesPinhole, RAM> mem;
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->as.handle;
    mem->ranges = ranges.raw();

    Memory<OptixSimulationDataRangesPinhole, VRAM_CUDA> d_mem;
    copy(mem, d_mem, m_stream);

    OptixProgramPtr program = m_programs[0];

    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream,
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataRangesPinhole ),
                &program->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }
}

Memory<float, VRAM_CUDA> PinholeSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
    Memory<float, VRAM_CUDA> res(m_width * m_height * Tbm.size());
    simulateRanges(Tbm, res);
    return res;
}

void PinholeSimulatorOptix::simulateNormals(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<Vector, VRAM_CUDA>& normals) const
{
    Memory<OptixSimulationDataNormalsPinhole, RAM> mem;
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->as.handle;
    mem->normals = normals.raw();

    Memory<OptixSimulationDataNormalsPinhole, VRAM_CUDA> d_mem;
    copy(mem, d_mem, m_stream);

    OptixProgramPtr program = m_programs[1];

    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream,
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataNormalsPinhole ),
                &program->sbt,
                m_width, // width Xdim
                m_height, // height Ydim
                Tbm.size() // depth Zdim
                ));
    } else {
        throw std::runtime_error("Return Bundle Combination not implemented for Optix Simulator");
    }
}

Memory<Vector, VRAM_CUDA> PinholeSimulatorOptix::simulateNormals(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
    Memory<Vector, VRAM_CUDA> res(m_width * m_height * Tbm.size());
    simulateNormals(Tbm, res);
    return res;
}



} // imagine