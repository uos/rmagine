#ifndef IMAGINE_SIMULATION_RESULTS_HPP
#define IMAGINE_SIMULATION_RESULTS_HPP

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

/**
 * @brief 
 * 
 * @tparam MemT 
 * 
 * Vector3d. 
 * x = phi
 * y = theta
 * z = range
 */
template<typename MemT>
struct ScanPoints {
    Memory<Point, MemT> scan_points;
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





struct MeshFace
{
    unsigned int mesh_id;
    unsigned int face_id;
};



} // namespace imagine

#endif // IMAGINE_SIMULATION_RESULTS_HPP