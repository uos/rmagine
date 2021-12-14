#ifndef IMAGINE_SIMULATION_RESULTS_HPP
#define IMAGINE_SIMULATION_RESULTS_HPP

#include <imagine/types/Bundle.hpp>
#include <imagine/types/Memory.hpp>
#include <imagine/math/types.h>

namespace imagine
{


template<typename MemT>
struct Hits {
    Memory<uint8_t, MemT> hits;
};

template<typename MemT>
struct Ranges {
    Memory<float, MemT> ranges;
};

template<typename MemT>
struct Points {
    Memory<Point, MemT> points;
};

template<typename MemT>
struct Normals {
    Memory<Vector, MemT> normals;
};

template<typename MemT>
struct FaceIds {
    Memory<unsigned int, MemT> face_ids;
};

template<typename MemT>
struct ObjectIds {
    Memory<unsigned int, MemT> object_ids;
};


template<typename MemT, typename BundleT>
void resizeMemoryBundle(BundleT& res, 
    unsigned int W,
    unsigned int H,
    unsigned int N )
{
    if constexpr(BundleT::template has<Hits<MemT> >())
    {
        res.Hits<MemT>::hits.resize(W*H*N);
    }

    if constexpr(BundleT::template has<Ranges<MemT> >())
    {
        res.Ranges<MemT>::ranges.resize(W*H*N);
    }

    if constexpr(BundleT::template has<Points<MemT> >())
    {
        res.Points<MemT>::points.resize(W*H*N);
    }

    if constexpr(BundleT::template has<Normals<MemT> >())
    {
        res.Normals<MemT>::normals.resize(W*H*N);
    }

    if constexpr(BundleT::template has<FaceIds<MemT> >())
    {
        res.FaceIds<MemT>::face_ids.resize(W*H*N);
    }

    if constexpr(BundleT::template has<ObjectIds<MemT> >())
    {
        res.ObjectIds<MemT>::object_ids.resize(W*H*N);
    }
}


} // namespace imagine

#endif // IMAGINE_SIMULATION_RESULTS_HPP