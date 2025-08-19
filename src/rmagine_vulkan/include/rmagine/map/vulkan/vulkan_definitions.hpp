#pragma once

#include <memory>
#include <rmagine/util/hashing.h>



namespace rmagine 
{

enum class VulkanGeometryType
{
    NONE,
    INSTANCE,
    MESH,
    POINTS //TODO: what is this for, do i need this, can i do this?
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
class AccelerationStructure;
class BottomLevelAccelerationStructure;
class TopLevelAccelerationStructure;



using VulkanEntityPtr = std::shared_ptr<VulkanEntity>;
using VulkanTransformablePtr = std::shared_ptr<VulkanTransformable>;
using VulkanGeometryPtr = std::shared_ptr<VulkanGeometry>;
using VulkanInstPtr = std::shared_ptr<VulkanInst>;
using VulkanScenePtr = std::shared_ptr<VulkanScene>;
using AccelerationStructurePtr = std::shared_ptr<AccelerationStructure>;
using BottomLevelAccelerationStructurePtr = std::shared_ptr<BottomLevelAccelerationStructure>;
using TopLevelAccelerationStructurePtr = std::shared_ptr<TopLevelAccelerationStructure>;

using VulkanEntityWPtr = std::weak_ptr<VulkanEntity>;
using VulkanTransformableWPtr = std::weak_ptr<VulkanTransformable>;
using VulkanGeometryWPtr = std::weak_ptr<VulkanGeometry>;
using VulkanInstWPtr = std::weak_ptr<VulkanInst>;
using VulkanSceneWPtr = std::weak_ptr<VulkanScene>;
using AccelerationStructureWPtr = std::weak_ptr<AccelerationStructure>;
using BottomLevelAccelerationStructureWPtr = std::weak_ptr<BottomLevelAccelerationStructure>;
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
