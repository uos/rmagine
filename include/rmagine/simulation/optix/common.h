#ifndef RMAGINE_SIMULATION_OPTIX_COMMON_H
#define RMAGINE_SIMULATION_OPTIX_COMMON_H


#include "OptixSimulationData.hpp"

#include <rmagine/simulation/SimulationResults.hpp>
#include <rmagine/types/MemoryCuda.hpp>


namespace rmagine
{

template<typename BundleT, typename ModelT>
static void setGenericData(
    BundleT& res, 
    OptixSimulationDataGeneric<ModelT>& mem)
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

template<typename BundleT, typename ModelT>
static void setGenericFlags(
    OptixSimulationDataGeneric<ModelT>& flags)
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


template<typename BundleT, typename ModelT>
static void setGenericFlags(
    const BundleT& res,
    OptixSimulationDataGeneric<ModelT>& flags)
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


}

#endif // RMAGINE_SIMULATION_OPTIX_COMMON_H