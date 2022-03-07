#include "rmagine/simulation/SphereSimulatorOptix.hpp"

#include "rmagine/simulation/optix/OptixSimulationData.hpp"
#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

// Scan Programs
#include <rmagine/simulation/optix/SphereProgramRanges.hpp>
#include <rmagine/simulation/optix/SphereProgramNormals.hpp>
#include <rmagine/simulation/optix/SphereProgramGeneric.hpp>

namespace rmagine
{

SphereSimulatorOptix::SphereSimulatorOptix()
:m_model(1)
,m_Tsb(1)
{
    CUDA_CHECK( cudaStreamCreate( &m_stream ) );
}

SphereSimulatorOptix::SphereSimulatorOptix(OptixMapPtr map)
:SphereSimulatorOptix()
{
    setMap(map);
}

SphereSimulatorOptix::~SphereSimulatorOptix()
{
    // std::cout << "Destruct SphereSimulatorOptix" << std::endl;
    cudaStreamDestroy(m_stream);
    
    m_programs.resize(0);
    m_generic_programs.clear();
}

void SphereSimulatorOptix::setMap(const OptixMapPtr map)
{
    m_map = map;
    // none generic version
    std::cout << "Set Map" << std::endl;
    
    m_programs.resize(2);
    std::cout << "- generate sphere program ranges with map" << std::endl;
    m_programs[0].reset(new SphereProgramRanges(map));
    std::cout << "- generate sphere program normals with map" << std::endl;
    m_programs[1].reset(new SphereProgramNormals(map));
}

void SphereSimulatorOptix::setTsb(const Memory<Transform, RAM>& Tsb)
{
    m_Tsb = Tsb;
}

void SphereSimulatorOptix::setTsb(const Transform& Tsb)
{
    Memory<Transform, RAM> tmp(1);
    tmp[0] = Tsb;
    setTsb(tmp);
}
    
void SphereSimulatorOptix::setModel(const Memory<SphericalModel, RAM>& model)
{
    m_width = model->getWidth();
    m_height = model->getHeight();
    m_model = model;
}

void SphereSimulatorOptix::setModel(const SphericalModel& model)
{
    Memory<SphericalModel, RAM> tmp(1);
    tmp[0] = model;
    setModel(tmp);
}

void SphereSimulatorOptix::simulateRanges(
    const Memory<Transform, VRAM_CUDA>& Tbm, 
    Memory<float, VRAM_CUDA>& ranges) const
{
    if(!m_map)
    {
        // no map set
        std::cout << "No Map assigned to Simulator!" << std::endl;
    }
    
    Memory<OptixSimulationDataRangesSphere, RAM> mem(1);
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->as.handle;
    mem->ranges = ranges.raw();

    Memory<OptixSimulationDataRangesSphere, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream);

    OptixProgramPtr program = m_programs[0];

    if(program)
    {
        std::cout << "LAUNCH " << m_width << "x" << m_height << "x" << Tbm.size() << std::endl;
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
        std::cout << "finished." << std::endl;
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
    Memory<OptixSimulationDataNormalsSphere, RAM> mem(1);
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->as.handle;
    mem->normals = normals.raw();

    Memory<OptixSimulationDataNormalsSphere, VRAM_CUDA> d_mem(1);
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

} // rmagine