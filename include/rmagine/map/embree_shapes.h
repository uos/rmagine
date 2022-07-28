#ifndef RMAGINE_MAP_EMBREE_SHAPES_H
#define RMAGINE_MAP_EMBREE_SHAPES_H

#include "EmbreeMesh.hpp"
#include "EmbreeDevice.hpp"
#include <memory>

namespace rmagine
{


class EmbreeSphere;
using EmbreeSpherePtr = std::shared_ptr<EmbreeSphere>;
using EmbreeSphereWPtr = std::weak_ptr<EmbreeSphere>;

class EmbreeSphere : public EmbreeMesh
{
    using Base = EmbreeMesh;
public:
    EmbreeSphere(
        float radius, 
        unsigned int num_long = 50, 
        unsigned int num_lat = 50,
        EmbreeDevicePtr device = embree_default_device()
    );
};

class EmbreeCube;
using EmbreeCubePtr = std::shared_ptr<EmbreeCube>;
using EmbreeCubeWPtr = std::weak_ptr<EmbreeCube>;

class EmbreeCube : public EmbreeMesh
{
    using Base = EmbreeMesh;
public:
    EmbreeCube(
        unsigned int side_triangles_exp = 1,
        EmbreeDevicePtr device = embree_default_device()
    );
};

class EmbreePlane;
using EmbreePlanePtr = std::shared_ptr<EmbreePlane>;
using EmbreePlaneWPtr = std::weak_ptr<EmbreePlane>;

class EmbreePlane : public EmbreeMesh
{
    using Base = EmbreeMesh;
public:
    EmbreePlane(
        unsigned int side_triangles_exp = 1,
        EmbreeDevicePtr device = embree_default_device()
    );
};


} // namespace rmagine


#endif // RMAGINE_MAP_EMBREE_SHAPES_H