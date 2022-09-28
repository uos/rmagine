#ifndef RMAGINE_MAP_OPTIX_SHAPES_H
#define RMAGINE_MAP_OPTIX_SHAPES_H

#include "optix_definitions.h"
#include "OptixMesh.hpp"

namespace rmagine
{

class OptixSphere : public OptixMesh 
{
public:
    using Base = OptixMesh;

    OptixSphere(unsigned int num_long = 50,
        unsigned int num_lat = 50,
        OptixContextPtr context = optix_default_context());

    virtual ~OptixSphere();
};

using OptixSpherePtr = std::shared_ptr<OptixSphere>;

class OptixCube : public OptixMesh 
{
public:
    using Base = OptixMesh;

    OptixCube(unsigned int side_triangles_exp = 1,
        OptixContextPtr context = optix_default_context());

    virtual ~OptixCube();
};

using OptixCubePtr = std::shared_ptr<OptixCube>;


class OptixPlane : public OptixMesh 
{
public:
    using Base = OptixMesh;

    OptixPlane(unsigned int side_triangles_exp = 1,
        OptixContextPtr context = optix_default_context());

    virtual ~OptixPlane();
};

using OptixPlanePtr = std::shared_ptr<OptixPlane>;
class OptixCylinder : public OptixMesh 
{
public:
    using Base = OptixMesh;

    OptixCylinder(unsigned int side_faces = 100,
        OptixContextPtr context = optix_default_context());

    virtual ~OptixCylinder();
};

using OptixCylinderPtr = std::shared_ptr<OptixCylinder>;


} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SHAPES_H