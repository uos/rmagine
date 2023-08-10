#ifndef RMAGINE_SIMULATION_OPTIX_COMMON_H
#define RMAGINE_SIMULATION_OPTIX_COMMON_H


#include "sim_program_data.h"

#include <rmagine/simulation/SimulationResults.hpp>
#include <rmagine/types/MemoryCuda.hpp>

#include <functional>


namespace rmagine
{

template<typename MemT, typename BundleT>
static void set_generic_data_(
    BundleT& res, 
    OptixSimulationDataGeneric& mem)
{
    //////////////////
    /// MemT
    if constexpr(BundleT::template has<Hits<MemT> >())
    {
        if(res.Hits<MemT>::hits.size() > 0)
        {
            mem.hits = res.Hits<MemT>::hits.raw();
        }
    }

    if constexpr(BundleT::template has<Ranges<MemT> >())
    {
        if(res.Ranges<MemT>::ranges.size() > 0)
        {
            mem.ranges = res.Ranges<MemT>::ranges.raw();
        }
    }

    if constexpr(BundleT::template has<Points<MemT> >())
    {
        if(res.Points<MemT>::points.size() > 0)
        {
            mem.points = res.Points<MemT>::points.raw();
        }
    }

    if constexpr(BundleT::template has<Normals<MemT> >())
    {
        if(res.Normals<MemT>::normals.size() > 0)
        {
            mem.normals = res.Normals<MemT>::normals.raw();
        }
    }

    if constexpr(BundleT::template has<FaceIds<MemT> >())
    {
        if(res.FaceIds<MemT>::face_ids.size() > 0)
        {
            mem.face_ids = res.FaceIds<MemT>::face_ids.raw();
        }
    }

    if constexpr(BundleT::template has<GeomIds<MemT> >())
    {
        if(res.GeomIds<MemT>::geom_ids.size() > 0)
        {
            mem.geom_ids = res.GeomIds<MemT>::geom_ids.raw();
        }
    }

    if constexpr(BundleT::template has<ObjectIds<MemT> >())
    {
        if(res.ObjectIds<MemT>::object_ids.size() > 0)
        {
            mem.object_ids = res.ObjectIds<MemT>::object_ids.raw();
        }
    }
}

template<typename BundleT>
static void set_generic_data(
    BundleT& res, 
    OptixSimulationDataGeneric& mem)
{
    set_generic_data_<VRAM_CUDA>(res, mem);
    set_generic_data_<UNIFIED_CUDA>(res, mem);
}

template<typename BundleT>
[[deprecated("Use set_generic_data() instead.")]]
static void setGenericData(
    BundleT& res, 
    OptixSimulationDataGeneric& mem)
{
    set_generic_data<BundleT>(res, mem);
}


template<typename MemT, typename BundleT>
static void set_generic_flags_(
    OptixSimulationDataGeneric& flags)
{
    if constexpr(BundleT::template has<Hits<MemT> >())
    {
        flags.computeHits = true;
    }

    if constexpr(BundleT::template has<Ranges<MemT> >())
    {
        flags.computeRanges = true;
    }

    if constexpr(BundleT::template has<Points<MemT> >())
    {
        flags.computePoints = true;
    }

    if constexpr(BundleT::template has<Normals<MemT> >())
    {
        flags.computeNormals = true;
    }

    if constexpr(BundleT::template has<FaceIds<MemT> >())
    {
        flags.computeFaceIds = true;
    }

    if constexpr(BundleT::template has<GeomIds<MemT> >())
    {
        flags.computeGeomIds = true;
    }

    if constexpr(BundleT::template has<ObjectIds<MemT> >())
    {
        flags.computeObjectIds = true;
    }
}

template<typename BundleT>
static void set_generic_flags(
    OptixSimulationDataGeneric& flags)
{
    set_generic_flags_<VRAM_CUDA, BundleT>(flags);
    set_generic_flags_<UNIFIED_CUDA, BundleT>(flags);
}

template<typename BundleT>
[[deprecated("Use set_generic_flags() instead.")]]
static void setGenericFlags(
    OptixSimulationDataGeneric& flags)
{
    set_generic_flags<BundleT>(flags);
}

template<typename MemT, typename BundleT>
static void set_generic_flags_(
    const BundleT& res,
    OptixSimulationDataGeneric& flags)
{
    // flags.computeHits = false;
    // flags.computeRanges = false;
    // flags.computePoints = false;
    // flags.computeNormals = false;
    // flags.computeFaceIds = false;
    // flags.computeGeomIds = false;
    // flags.computeObjectIds = false;

    if constexpr(BundleT::template has<Hits<MemT> >())
    {
        if(res.Hits<MemT>::hits.size() > 0)
        {
            flags.computeHits = true;
        }
    }

    if constexpr(BundleT::template has<Ranges<MemT> >())
    {
        if(res.Ranges<MemT>::ranges.size() > 0)
        {
            flags.computeRanges = true;
        }
    }

    if constexpr(BundleT::template has<Points<MemT> >())
    {
        if(res.Points<MemT>::points.size() > 0)
        {
            flags.computePoints = true;
        }
    }

    if constexpr(BundleT::template has<Normals<MemT> >())
    {
        if(res.Normals<MemT>::normals.size() > 0)
        {
            flags.computeNormals = true;
        }
    }

    if constexpr(BundleT::template has<FaceIds<MemT> >())
    {
        if(res.FaceIds<MemT>::face_ids.size() > 0)
        {
            flags.computeFaceIds = true;
        }
    }

    if constexpr(BundleT::template has<GeomIds<MemT> >())
    {
        if(res.GeomIds<MemT>::geom_ids.size() > 0)
        {
            flags.computeGeomIds = true;
        }
    }

    if constexpr(BundleT::template has<ObjectIds<MemT> >())
    {
        if(res.ObjectIds<MemT>::object_ids.size() > 0)
        {
            flags.computeObjectIds = true;
        }
    }
}

template<typename BundleT>
static void set_generic_flags(
    const BundleT& res,
    OptixSimulationDataGeneric& flags)
{
    set_generic_flags_<VRAM_CUDA>(res, flags);
    set_generic_flags_<UNIFIED_CUDA>(res, flags);
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