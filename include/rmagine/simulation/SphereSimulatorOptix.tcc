
#include <rmagine/simulation/optix/SphereProgramGeneric.hpp>
#include <rmagine/util/optix/OptixDebug.hpp>
#include <optix.h>
#include <optix_stubs.h>

// #include <rmagine/util/StopWatch.hpp>
#include <rmagine/map/optix/OptixScene.hpp>

namespace rmagine
{

template<typename BundleT>
void setGenericData(
    BundleT& res, 
    OptixSimulationDataGenericSphere& mem)
{
    if constexpr(BundleT::template has<Hits<VRAM_CUDA> >())
    {
        mem.hits = res.Hits<VRAM_CUDA>::hits.raw();
    }

    if constexpr(BundleT::template has<Ranges<VRAM_CUDA> >())
    {
        mem.ranges = res.Ranges<VRAM_CUDA>::ranges.raw();
    }

    if constexpr(BundleT::template has<Points<VRAM_CUDA> >())
    {
        mem.points = res.Points<VRAM_CUDA>::points.raw();
    }

    if constexpr(BundleT::template has<Normals<VRAM_CUDA> >())
    {
        mem.normals = res.Normals<VRAM_CUDA>::normals.raw();
    }

    if constexpr(BundleT::template has<FaceIds<VRAM_CUDA> >())
    {
        mem.face_ids = res.FaceIds<VRAM_CUDA>::face_ids.raw();
    }

    if constexpr(BundleT::template has<ObjectIds<VRAM_CUDA> >())
    {
        mem.object_ids = res.ObjectIds<VRAM_CUDA>::object_ids.raw();
    }
}

template<typename BundleT>
void setGenericFlags(
    OptixSimulationDataGenericSphere& flags)
{
    flags.computeHits = false;
    flags.computeRanges = false;
    flags.computePoints = false;
    flags.computeNormals = false;
    flags.computeFaceIds = false;
    flags.computeObjectIds = false;

    if constexpr(BundleT::template has<Hits<VRAM_CUDA> >())
    {
        flags.computeHits = true;
    }

    if constexpr(BundleT::template has<Ranges<VRAM_CUDA> >())
    {
        flags.computeRanges = true;
    }

    if constexpr(BundleT::template has<Points<VRAM_CUDA> >())
    {
        flags.computePoints = true;
    }

    if constexpr(BundleT::template has<Normals<VRAM_CUDA> >())
    {
        flags.computeNormals = true;
    }

    if constexpr(BundleT::template has<FaceIds<VRAM_CUDA> >())
    {
        flags.computeFaceIds = true;
    }

    if constexpr(BundleT::template has<ObjectIds<VRAM_CUDA> >())
    {
        flags.computeObjectIds = true;
    }
}

template<typename BundleT>
void SphereSimulatorOptix::preBuildProgram()
{
    OptixSimulationDataGenericSphere flags;
    setGenericFlags<BundleT>(flags);
    auto it = m_generic_programs.find(flags);
    
    if(it == m_generic_programs.end())
    {
        OptixProgramPtr program;
        if(m_map)
        {
            program = std::make_shared<SphereProgramGeneric>(m_map, flags);
        }
        m_generic_programs[flags] = program;
    }
}

template<typename BundleT>
void SphereSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    BundleT& res)
{
    Memory<OptixSimulationDataGenericSphere, RAM> mem(1);

    setGenericFlags<BundleT>(mem[0]);

    auto it = m_generic_programs.find(mem[0]);
    OptixProgramPtr program;
    if(it == m_generic_programs.end())
    {
        if(m_map)
        {
            program = std::make_shared<SphereProgramGeneric>(m_map, mem[0]);
        }
        m_generic_programs[mem[0]] = program;
    } else {
        program = it->second;
        program->updateSBT();
    }

    // set general data
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();

    if(m_map)
    {
        mem->handle = m_map->scene()->getRoot()->acc()->handle;
    }
    

    // set generic data
    setGenericData(res, mem[0]);

    // 10000 velodynes 
    // - upload Params: 0.000602865s
    // - launch: 5.9642e-05s
    // => this takes too long. Can we somehow preupload stuff?
    Memory<OptixSimulationDataGenericSphere, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream);
    
    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream,
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataGenericSphere ),
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
BundleT SphereSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm)
{
    BundleT res;
    resizeMemoryBundle<VRAM_CUDA>(res, m_width, m_height, Tbm.size());
    simulate(Tbm, res);
    return res;
}

} // namespace rmagine