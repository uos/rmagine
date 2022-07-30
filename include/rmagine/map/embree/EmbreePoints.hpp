#ifndef RMAGINE_MAP_EMBREE_POINTS_HPP
#define RMAGINE_MAP_EMBREE_POINTS_HPP

#include "embree_types.h"

#include <rmagine/types/Memory.hpp>
#include <assimp/mesh.h>

#include <rmagine/math/types.h>
#include <rmagine/types/mesh_types.h>

#include <memory>
#include <embree3/rtcore.h>
#include "EmbreeDevice.hpp"
#include "EmbreeGeometry.hpp"


namespace rmagine
{

struct PointWithRadius
{
    Vector3 p;
    float r;
};

class EmbreePoints
: public EmbreeGeometry
{
public:
    using Base = EmbreeGeometry;
    EmbreePoints(EmbreeDevicePtr device = embree_default_device());
    EmbreePoints(unsigned int Npoints, EmbreeDevicePtr device = embree_default_device());

    virtual ~EmbreePoints();

    void init(unsigned int Npoints);

    unsigned int Npoints;
    PointWithRadius* points;
};

// TODO
// class EmbreePointDiscs
// {

// };


} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_POINT_HPP