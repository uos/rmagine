#include "rmagine/simulation/SphereSimulatorOptix.hpp"

#include "rmagine/simulation/optix/OptixSimulationData.hpp"
#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <rmagine/map/optix/OptixInstances.hpp>

// Scan Programs
#include <rmagine/simulation/optix/SphereProgramRanges.hpp>
#include <rmagine/simulation/optix/SphereProgramNormals.hpp>
#include <rmagine/simulation/optix/SphereProgramGeneric.hpp>

#include <rmagine/util/cuda/CudaStream.hpp>

namespace rmagine
{

SphereSimulatorOptix::SphereSimulatorOptix()
:m_model(1)
,m_Tsb(1)
{
    Memory<Transform, RAM_CUDA> I(1);
    I->setIdentity();
    m_Tsb = I;
}

SphereSimulatorOptix::SphereSimulatorOptix(OptixMapPtr map)
:SphereSimulatorOptix()
{
    setMap(map);
}

SphereSimulatorOptix::SphereSimulatorOptix(OptixScenePtr scene)
:SphereSimulatorOptix()
{
    OptixMapPtr map = std::make_shared<OptixMap>(scene);
    setMap(map);
}

SphereSimulatorOptix::~SphereSimulatorOptix()
{
    // std::cout << "[SphereSimulatorOptix] ~SphereSimulatorOptix" << std::endl;
    m_programs.resize(0);
    m_generic_programs.clear();
}

void SphereSimulatorOptix::setMap(const OptixMapPtr map)
{
    std::cout << "[SphereSimulatorOptix::setMap] done." << std::endl;
    m_map = map;
    // none generic version
    m_programs.resize(2);
    m_programs[0] = std::make_shared<SphereProgramRanges>(map);
    m_programs[1] = std::make_shared<SphereProgramNormals>(map);

    // m_stream = m_map->context()->getCudaContext()->createStream();
    m_stream = m_map->stream();

    // need to create stream after map was created: cuda device api context is required
    // CUDA_CHECK( cudaStreamCreate( &m_stream ) );
    std::cout << "[SphereSimulatorOptix::setMap] done." << std::endl;
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
        throw std::runtime_error("[SphereSimulatorOptix] simulateRanges(): No Map available!");
        return;
    }

    auto optix_ctx = m_map->context();
    auto cuda_ctx = optix_ctx->getCudaContext();
    if(!cuda_ctx->isActive())
    {
        std::cout << "[SphereSimulatorOptix::simulateRanges() Need to activate map context" << std::endl;
        cuda_ctx->use();
    }

    Memory<OptixSimulationDataRangesSphere, RAM> mem(1);
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->ranges = ranges.raw();
    mem->handle = m_map->scene()->getRoot()->acc()->handle;
    

    Memory<OptixSimulationDataRangesSphere, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream->handle());

    OptixProgramPtr program = m_programs[0];

    if(program)
    {
        // std::cout << "Simulating " << Tbm.size() << " SphericalSensors " << m_width << "x" << m_height << std::endl;
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream->handle(),
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
    if(!m_map)
    {
        // no map set
        throw std::runtime_error("[SphereSimulatorOptix] simulateRanges(): No Map available!");
        return;
    }

    auto optix_ctx = m_map->context();
    auto cuda_ctx = optix_ctx->getCudaContext();
    if(!cuda_ctx->isActive())
    {
        std::cout << "[SphereSimulatorOptix::simulateRanges() Need to activate map context" << std::endl;
        cuda_ctx->use();
    }

    Memory<OptixSimulationDataNormalsSphere, RAM> mem(1);
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();
    mem->handle = m_map->scene()->getRoot()->acc()->handle;
    mem->normals = normals.raw();

    Memory<OptixSimulationDataNormalsSphere, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream->handle());

    OptixProgramPtr program = m_programs[1];

    if(program)
    {
        program->updateSBT();
        
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream->handle(),
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