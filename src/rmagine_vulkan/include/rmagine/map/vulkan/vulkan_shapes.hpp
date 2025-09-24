#pragma once

#include "VulkanMesh.hpp"
#include <rmagine/util/synthetic.h>



namespace rmagine
{

class VulkanSphere : public VulkanMesh 
{
public:
    using Base = VulkanMesh;

    VulkanSphere(unsigned int num_long = 50, unsigned int num_lat = 50);

    virtual ~VulkanSphere();
};
using VulkanSpherePtr = std::shared_ptr<VulkanSphere>;



class VulkanCube : public VulkanMesh 
{
public:
    using Base = VulkanMesh;

    VulkanCube(unsigned int side_triangles_exp = 1);

    virtual ~VulkanCube();
};
using VulkanCubePtr = std::shared_ptr<VulkanCube>;



class VulkanPlane : public VulkanMesh 
{
public:
    using Base = VulkanMesh;

    VulkanPlane(unsigned int side_triangles_exp = 1);

    virtual ~VulkanPlane();
};
using VulkanPlanePtr = std::shared_ptr<VulkanPlane>;



class VulkanCylinder : public VulkanMesh 
{
public:
    using Base = VulkanMesh;

    VulkanCylinder(unsigned int side_faces = 100);

    virtual ~VulkanCylinder();
};
using VulkanCylinderPtr = std::shared_ptr<VulkanCylinder>;

} // namespace rmagine
