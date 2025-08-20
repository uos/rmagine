#pragma once

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <unordered_set>
#include <optional>

#include <vulkan/vulkan.h>

#include <rmagine/map/AssimpIO.hpp>
#include <rmagine/util/VulkanContext.hpp>
#include <rmagine/util/VulkanUtil.hpp>
#include <rmagine/util/IDGen.hpp>
#include "accelerationStructure/AccelerationStructure.hpp"
#include "VulkanMesh.hpp"
#include "VulkanEntity.hpp"
#include "VulkanGeometry.hpp"
#include "VulkanTransformable.hpp"
#include "VulkanInst.hpp"



namespace rmagine
{

class VulkanScene : public VulkanEntity
{
private:
    AccelerationStructurePtr m_as = nullptr;

    VulkanSceneType m_type = VulkanSceneType::NONE;
    VulkanGeometryType m_geom_type = VulkanGeometryType::MESH;

    IDGen gen;

    std::map<unsigned int, VulkanGeometryPtr> m_geometries;
    std::unordered_map<VulkanGeometryPtr, unsigned int> m_ids;

    std::unordered_set<VulkanInstWPtr> m_parents;

    bool m_geom_added = false;
    bool m_geom_removed = false;

    // filled after commit
    unsigned int m_depth = 0;

    //only used for top level as
    Memory<VkAccelerationStructureInstanceKHR, RAM> m_asInstances_ram;
    Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL> m_asInstances;
    // std::unordered_set<VulkanMeshWPtr> m_meshes;

public:
    //TODO: TEMP; FIX LATER
    Memory<Point, VULKAN_DEVICE_LOCAL>* vertexptr = nullptr;
    Memory<Face, VULKAN_DEVICE_LOCAL>* indexptr = nullptr;

    VulkanScene();

    virtual ~VulkanScene();


    unsigned int add(VulkanGeometryPtr geom);
    unsigned int get(VulkanGeometryPtr geom) const;
    std::optional<unsigned int> getOpt(VulkanGeometryPtr geom) const;
    bool has(VulkanGeometryPtr geom) const;
    bool has(unsigned int geom_id) const;
    bool remove(VulkanGeometryPtr geom);
    VulkanGeometryPtr remove(unsigned int geom_id);

    std::map<unsigned int, VulkanGeometryPtr> geometries() const;
    std::unordered_map<VulkanGeometryPtr, unsigned int> ids() const;
    
    VulkanInstPtr instantiate();

    
    inline VulkanSceneType type() const 
    {
        return m_type;
    }

    inline VulkanGeometryType geom_type() const
    {
        return m_geom_type;
    }

    // geometry can be instanced
    void cleanupParents();
    std::unordered_set<VulkanInstPtr> parents() const;
    void addParent(VulkanInstPtr parent);

    /**
     * @brief Call commit after the scene was filles with
     * geometries or instances to begin the building/updating process
     * of the acceleration structure
     * - only after commit it is possible to raytrace
     */
    void commit();

    // ACCASSIBLE AFTER COMMIT
    inline AccelerationStructurePtr as() const
    {
        return m_as;
    }

    inline unsigned int depth() const 
    {
        return m_depth;
    }

    size_t numOfChildNodes() const
    {
        return m_geometries.size();
    }

    Memory<VkAccelerationStructureInstanceKHR, VULKAN_DEVICE_LOCAL>& getASInstances()
    {
        return m_asInstances;
    }
};

using VulkanScenePtr = std::shared_ptr<VulkanScene>;



VulkanScenePtr make_vulkan_scene(Memory<Point, RAM>& vertices_ram, Memory<Face, RAM>& faces_ram);

VulkanScenePtr make_vulkan_scene(const aiScene* ascene);

} // namespace rmagine
