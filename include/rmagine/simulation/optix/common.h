#ifndef RMAGINE_SIMULATION_OPTIX_COMMON_H
#define RMAGINE_SIMULATION_OPTIX_COMMON_H


#include "sim_program_data.h"

#include <rmagine/simulation/SimulationResults.hpp>
#include <rmagine/types/MemoryCuda.hpp>

#include <functional>


namespace rmagine
{

template<typename BundleT>
static void set_generic_data(
    BundleT& res, 
    OptixSimulationDataGeneric& mem)
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
[[deprecated("Use set_generic_data() instead.")]]
static void setGenericData(
    BundleT& res, 
    OptixSimulationDataGeneric& mem)
{
    set_generic_data<BundleT>(res, mem);
}

template<typename BundleT>
static void set_generic_flags(
    OptixSimulationDataGeneric& flags)
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
[[deprecated("Use set_generic_flags() instead.")]]
static void setGenericFlags(
    OptixSimulationDataGeneric& flags)
{
    set_generic_flags<BundleT>(flags);
}

template<typename BundleT>
static void set_generic_flags(
    const BundleT& res,
    OptixSimulationDataGeneric& flags)
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
[[deprecated("Use set_generic_flags() instead.")]]
static void setGenericFlags(
    const BundleT& res,
    OptixSimulationDataGeneric& flags)
{
    set_generic_flags<BundleT>(res, flags);
}


} // namespace rmagine




namespace std {

template<>
struct hash<rmagine::OptixSimulationDataGeneric>
{
    std::size_t operator()(const rmagine::OptixSimulationDataGeneric& k) const
    {
        // first 24 bits are reserved for bool flags (bound values)
        std::size_t hitsKey = static_cast<std::size_t>(k.computeHits) << 0;
        std::size_t rangesKey = static_cast<std::size_t>(k.computeRanges) << 1;
        std::size_t pointKey = static_cast<std::size_t>(k.computePoints) << 2;
        std::size_t normalsKey = static_cast<std::size_t>(k.computeNormals) << 3;
        std::size_t faceIdsKey = static_cast<std::size_t>(k.computeFaceIds) << 4;
        std::size_t geomIdsKey = static_cast<std::size_t>(k.computeGeomIds) << 5;
        std::size_t objectIdsKey = static_cast<std::size_t>(k.computeObjectIds) << 6;

        // next 8 bit are reserved for sensor type
        // sensor_type should not be higher than 2**8=256
        std::size_t sensorTypeKey = static_cast<std::size_t>(k.model_type) << 24;
        
        // bitwise or
        return (hitsKey | rangesKey | pointKey | normalsKey | faceIdsKey | geomIdsKey | objectIdsKey);
    }
};

} // namespace std

namespace rmagine
{

inline bool operator==(const OptixSimulationDataGeneric &a, const OptixSimulationDataGeneric &b)
{ 
    return std::hash<OptixSimulationDataGeneric>()(a) == std::hash<OptixSimulationDataGeneric>()(b);
}

} // namespace rmagine



#endif // RMAGINE_SIMULATION_OPTIX_COMMON_H