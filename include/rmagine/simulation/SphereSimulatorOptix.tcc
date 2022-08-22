
#include <rmagine/simulation/optix/SphereProgramGeneric.hpp>
#include <rmagine/util/optix/OptixDebug.hpp>
#include <optix.h>
#include <optix_stubs.h>

// #include <rmagine/util/StopWatch.hpp>
#include <rmagine/map/optix/OptixScene.hpp>
#include <rmagine/util/cuda/CudaStream.hpp>

namespace rmagine
{

template<typename BundleT>
void setGenericData(
    BundleT& res, 
    OptixSimulationDataGenericSphere& mem)
{
    if constexpr(BundleT::template has<Hits<VRAM_CUDA> >())
    {
        if(res.Hits<VRAM_CUDA>::hits.size() > 0)
        {
            mem.hits = res.Hits<VRAM_CUDA>::hits.raw();
        }
    }

    if constexpr(BundleT::template has<Ranges<VRAM_CUDA> >())
    {
        if(res.Ranges<VRAM_CUDA>::ranges.size() > 0)
        {
            mem.ranges = res.Ranges<VRAM_CUDA>::ranges.raw();
        }
    }

    if constexpr(BundleT::template has<Points<VRAM_CUDA> >())
    {
        if(res.Points<VRAM_CUDA>::points.size() > 0)
        {
            mem.points = res.Points<VRAM_CUDA>::points.raw();
        }
    }

    if constexpr(BundleT::template has<Normals<VRAM_CUDA> >())
    {
        if(res.Normals<VRAM_CUDA>::normals.size() > 0)
        {
            mem.normals = res.Normals<VRAM_CUDA>::normals.raw();
        }
    }

    if constexpr(BundleT::template has<FaceIds<VRAM_CUDA> >())
    {
        if(res.FaceIds<VRAM_CUDA>::face_ids.size() > 0)
        {
            mem.face_ids = res.FaceIds<VRAM_CUDA>::face_ids.raw();
        }
    }

    if constexpr(BundleT::template has<GeomIds<VRAM_CUDA> >())
    {
        if(res.GeomIds<VRAM_CUDA>::geom_ids.size() > 0)
        {
            mem.geom_ids = res.GeomIds<VRAM_CUDA>::geom_ids.raw();
        }
    }

    if constexpr(BundleT::template has<ObjectIds<VRAM_CUDA> >())
    {
        if(res.ObjectIds<VRAM_CUDA>::object_ids.size() > 0)
        {
            mem.object_ids = res.ObjectIds<VRAM_CUDA>::object_ids.raw();
        }
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
    flags.computeGeomIds = false;
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

    if constexpr(BundleT::template has<GeomIds<VRAM_CUDA> >())
    {
        flags.computeGeomIds = true;
    }

    if constexpr(BundleT::template has<ObjectIds<VRAM_CUDA> >())
    {
        flags.computeObjectIds = true;
    }
}


template<typename BundleT>
void setGenericFlags(
    const BundleT& res,
    OptixSimulationDataGenericSphere& flags)
{
    flags.computeHits = false;
    flags.computeRanges = false;
    flags.computePoints = false;
    flags.computeNormals = false;
    flags.computeFaceIds = false;
    flags.computeGeomIds = false;
    flags.computeObjectIds = false;

    if constexpr(BundleT::template has<Hits<VRAM_CUDA> >())
    {
        if(res.hits.size() > 0)
        {
            flags.computeHits = true;
        }
    }

    if constexpr(BundleT::template has<Ranges<VRAM_CUDA> >())
    {
        if(res.ranges.size() > 0)
        {
            flags.computeRanges = true;
        }
    }

    if constexpr(BundleT::template has<Points<VRAM_CUDA> >())
    {
        if(res.points.size() > 0)
        {
            flags.computePoints = true;
        }
    }

    if constexpr(BundleT::template has<Normals<VRAM_CUDA> >())
    {
        if(res.normals.size() > 0)
        {
            flags.computeNormals = true;
        }
    }

    if constexpr(BundleT::template has<FaceIds<VRAM_CUDA> >())
    {
        if(res.face_ids.size() > 0)
        {
            flags.computeFaceIds = true;
        }
    }

    if constexpr(BundleT::template has<GeomIds<VRAM_CUDA> >())
    {
        if(res.geom_ids.size() > 0)
        {
            flags.computeGeomIds = true;
        }
    }

    if constexpr(BundleT::template has<ObjectIds<VRAM_CUDA> >())
    {
        if(res.object_ids.size() > 0)
        {
            flags.computeObjectIds = true;
        }
    }
}

template<typename BundleT>
void SphereSimulatorOptix::preBuildProgram()
{
    if(!m_map)
    {
        throw std::runtime_error("[SphereSimulatorOptix] preBuildProgram(): No Map available!");
    }

    OptixSimulationDataGenericSphere flags;
    setGenericFlags<BundleT>(flags);
    auto it = m_generic_programs.find(flags);
    
    if(it == m_generic_programs.end())
    {
        m_generic_programs[flags] = std::make_shared<SphereProgramGeneric>(m_map, flags);
    }
}

template<typename BundleT>
void SphereSimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    BundleT& res)
{
    if(!m_map)
    {
        // no map set
        throw std::runtime_error("[SphereSimulatorOptix] simulate(): No Map available!");
        return;
    }

    auto optix_ctx = m_map->context();
    auto cuda_ctx = optix_ctx->getCudaContext();
    if(!cuda_ctx->isActive())
    {
        std::cout << "[SphereSimulatorOptix::simulate() Need to activate map context" << std::endl;
        cuda_ctx->use();
    }

    Memory<OptixSimulationDataGenericSphere, RAM> mem(1);

    setGenericFlags(res, mem[0]);

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
        // program->updateSBT();
    }

    // set general data
    mem->Tsb = m_Tsb.raw();
    mem->model = m_model.raw();
    mem->Tbm = Tbm.raw();

    mem->handle = m_map->scene()->as()->handle;
    
    // set generic data
    setGenericData(res, mem[0]);

    // 10000 velodynes 
    // - upload Params: 0.000602865s
    // - launch: 5.9642e-05s
    // => this takes too long. Can we somehow preupload stuff?
    Memory<OptixSimulationDataGenericSphere, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, m_stream->handle());

    
    if(program)
    {
        OPTIX_CHECK( optixLaunch(
                program->pipeline,
                m_stream->handle(),
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