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
#include <rmagine/util/VulkanContextUtil.hpp>
#include <rmagine/util/IDGen.hpp>
#include <rmagine/util/assimp/helper.h>
#include "vulkan_definitions.hpp"
#include "accelerationStructure/AccelerationStructure.hpp"
#include "accelerationStructure/TopLevelAccelerationStructure.hpp"
#include "accelerationStructure/BottomLevelAccelerationStructure.hpp"
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

public:
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

    /**
     * get the size of the complete acceleration structure in bytes
     * only works correctly for scenes with a deph of at most 2 (deeper scenes are currently not supported anyways)
     */
    size_t getAsSize() const
    {
        if(m_type == VulkanSceneType::GEOMETRIES)
        {
            return m_as->getSize();//get the size of the bottom level acceleration structure
        }
        else if(m_type == VulkanSceneType::INSTANCES)
        {
            size_t size = m_as->getSize();//get the size of the top level acceleration structure

            //get the size of all the bottom level acceleration structures
            std::unordered_set<VulkanScenePtr> meshScenes;
            for(auto const& geometry : m_geometries)
            {
                auto inst = geometry.second->this_shared<VulkanInst>();
                meshScenes.insert(inst->scene());
            }

            for(auto const& meshScene : meshScenes)
            {
                size += meshScene->getAsSize();
            }

            return size;
        }
        else
        {
            return 0;
        }
    }
};

using VulkanScenePtr = std::shared_ptr<VulkanScene>;



/**
 * creates a simple vulkan scene from some verticies and indicies/faces
 * 
 * @param vertices_ram verticies
 * 
 * @param faces_ram indicies/faces
 * 
 * @return vulkan scene
 */
VulkanScenePtr make_vulkan_scene(Memory<Point, RAM>& vertices_ram, Memory<Face, RAM>& faces_ram);

/**
 * creates a vulkan scene from an Assimp scene
 * 
 * @param meshfile Assimp scene
 * 
 * @return vulkan scene
 */
VulkanScenePtr make_vulkan_scene(const aiScene* ascene);

} // namespace rmagine
