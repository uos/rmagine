#pragma once

#include <memory>
#include <rmagine/util/hashing.h>



//mostly taken from "optix_definitions.h"



namespace rmagine 
{

enum class VulkanGeometryType
{
    NONE,
    INSTANCE,
    MESH,
    POINTS
};

enum class VulkanSceneType
{
    NONE,
    INSTANCES,
    GEOMETRIES
};



class VulkanEntity;
class VulkanTransformable;
class VulkanGeometry;
class VulkanInst;
class VulkanMesh;
class VulkanScene;
class BottomLevelAccelerationStructure;
class BottomLevelGeometryInstance;
class TopLevelAccelerationStructure;



using VulkanEntityPtr = std::shared_ptr<VulkanEntity>;
using VulkanTransformablePtr = std::shared_ptr<VulkanTransformable>;
using VulkanGeometryPtr = std::shared_ptr<VulkanGeometry>;
using VulkanInstPtr = std::shared_ptr<VulkanInst>;
using VulkanScenePtr = std::shared_ptr<VulkanScene>;
using BottomLevelAccelerationStructurePtr = std::shared_ptr<BottomLevelAccelerationStructure>;
using BottomLevelGeometryInstancePtr = std::shared_ptr<BottomLevelGeometryInstance>;
using TopLevelAccelerationStructurePtr = std::shared_ptr<TopLevelAccelerationStructure>;

using VulkanEntityWPtr = std::weak_ptr<VulkanEntity>;
using VulkanTransformableWPtr = std::weak_ptr<VulkanTransformable>;
using VulkanGeometryWPtr = std::weak_ptr<VulkanGeometry>;
using VulkanInstWPtr = std::weak_ptr<VulkanInst>;
using VulkanSceneWPtr = std::weak_ptr<VulkanScene>;
using BottomLevelAccelerationStructureWPtr = std::weak_ptr<BottomLevelAccelerationStructure>;
using BottomLevelGeometryInstanceWPtr = std::weak_ptr<BottomLevelGeometryInstance>;
using TopLevelAccelerationStructureWPtr = std::weak_ptr<TopLevelAccelerationStructure>;

} // namespace rmagine



namespace std
{

// INSTANCE
template<>
struct hash<rmagine::VulkanInstWPtr> 
    : public rmagine::weak_hash<rmagine::VulkanInst>
{};

template<>
struct equal_to<rmagine::VulkanInstWPtr> 
    : public rmagine::weak_equal_to<rmagine::VulkanInst>
{};

template<>
struct less<rmagine::VulkanInstWPtr> 
    : public rmagine::weak_less<rmagine::VulkanInst>
{};



// SCENE
template<>
struct hash<rmagine::VulkanSceneWPtr> 
    : public rmagine::weak_hash<rmagine::VulkanScene>
{};

template<>
struct equal_to<rmagine::VulkanSceneWPtr> 
    : public rmagine::weak_equal_to<rmagine::VulkanScene>
{};

template<>
struct less<rmagine::VulkanSceneWPtr> 
    : public rmagine::weak_less<rmagine::VulkanScene>
{};

} // namespace std
