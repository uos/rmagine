#ifndef RMAGINE_SIMULATION_EMBREE_COMMON_H
#define RMAGINE_SIMULATION_EMBREE_COMMON_H


#include <rmagine/types/Memory.hpp>
// ?
// #include <rmagine/types/MemoryCuda.hpp>

namespace rmagine
{

struct SimulationFlags 
{
    bool hits;
    bool ranges;
    bool points;
    bool normals;
    bool object_ids;
    bool geom_ids;
    bool face_ids;

    static SimulationFlags Zero()
    {
        SimulationFlags flags;

        flags.hits = false;
        flags.ranges = false;
        flags.points = false;
        flags.normals = false;
        flags.object_ids = false;
        flags.geom_ids = false;
        flags.face_ids = false;

        return flags;
    }
};

template<typename MemT, typename BundleT>
static void set_simulation_flags_(
    SimulationFlags& flags)
{
    if constexpr(BundleT::template has<Hits<MemT> >())
    {
        flags.hits = true;
    }

    if constexpr(BundleT::template has<Ranges<MemT> >())
    {
        flags.ranges = true;
    }

    if constexpr(BundleT::template has<Points<MemT> >())
    {
        flags.points = true;
    }

    if constexpr(BundleT::template has<Normals<MemT> >())
    {
        flags.normals = true;
    }

    if constexpr(BundleT::template has<FaceIds<MemT> >())
    {
        flags.face_ids = true;
    }

    if constexpr(BundleT::template has<GeomIds<MemT> >())
    {
        flags.geom_ids = true;
    }

    if constexpr(BundleT::template has<ObjectIds<MemT> >())
    {
        flags.object_ids = true;
    }
}


template<typename MemT, typename BundleT>
static void set_simulation_flags_(
    const BundleT& res,
    SimulationFlags& flags)
{
    if constexpr(BundleT::template has<Hits<MemT> >())
    {
        if(res.Hits<MemT>::hits.size() > 0)
        {
            flags.hits = true;
        }
    }

    if constexpr(BundleT::template has<Ranges<MemT> >())
    {
        if(res.Ranges<MemT>::ranges.size() > 0)
        {
            flags.ranges = true;
        }
    }

    if constexpr(BundleT::template has<Points<MemT> >())
    {
        if(res.Points<MemT>::points.size() > 0)
        {
            flags.points = true;
        }
    }

    if constexpr(BundleT::template has<Normals<MemT> >())
    {
        if(res.Normals<MemT>::normals.size() > 0)
        {
            flags.normals = true;
        }
    }

    if constexpr(BundleT::template has<FaceIds<MemT> >())
    {
        if(res.FaceIds<MemT>::face_ids.size() > 0)
        {
            flags.face_ids = true;
        }
    }

    if constexpr(BundleT::template has<GeomIds<MemT> >())
    {
        if(res.GeomIds<MemT>::geom_ids.size() > 0)
        {
            flags.geom_ids = true;
        }
    }

    if constexpr(BundleT::template has<ObjectIds<MemT> >())
    {
        if(res.ObjectIds<MemT>::object_ids.size() > 0)
        {
            flags.object_ids = true;
        }
    }
}



} // namespace rmagine


#endif // RMAGINE_SIMULATION_EMBREE_COMMON_H