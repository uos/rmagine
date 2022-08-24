#ifndef RMAGINE_MAP_OPTIX_SCENE_HPP
#define RMAGINE_MAP_OPTIX_SCENE_HPP

#include <rmagine/util/optix/OptixContext.hpp>
#include <rmagine/util/IDGen.hpp>

#include <optix.h>

#include "optix_definitions.h"
#include "optix_sbt.h"

#include "OptixEntity.hpp"

#include <map>

#include <assimp/scene.h>

#include <rmagine/types/MemoryCuda.hpp>
#include <unordered_set>


namespace rmagine
{

struct OptixSceneCommitResult
{
    bool depth_changed = false;
    bool sbt_size_changed = false;
};

class OptixScene 
: public OptixEntity
{
public:
    

    OptixScene(OptixContextPtr context = optix_default_context());

    virtual ~OptixScene();

    unsigned int add(OptixGeometryPtr geom);
    unsigned int get(OptixGeometryPtr geom) const;
    std::optional<unsigned int> getOpt(OptixGeometryPtr geom) const;
    bool has(OptixGeometryPtr geom) const;
    bool has(unsigned int geom_id) const;
    bool remove(OptixGeometryPtr geom);
    OptixGeometryPtr remove(unsigned int geom_id);

    std::map<unsigned int, OptixGeometryPtr> geometries() const;
    std::unordered_map<OptixGeometryPtr, unsigned int> ids() const;
    
    OptixInstPtr instantiate();

    inline OptixSceneType type() const 
    {
        return m_type;
    }

    inline OptixGeometryType geom_type() const
    {
        return m_geom_type;
    }

    // geometry can be instanced
    void cleanupParents();
    std::unordered_set<OptixInstPtr> parents() const;
    void addParent(OptixInstPtr parent);

    /**
     * @brief Call commit after the scene was filles with
     * geometries or instances to begin the building/updating process
     * of the acceleration structure
     * - only after commit it is possible to raytrace
     * 
     */
    OptixSceneCommitResult commit();

    // ACCASSIBLE AFTER COMMIT
    inline OptixAccelerationStructurePtr as() const
    {
        return m_as;
    }

    inline unsigned int traversableGraphFlags() const
    {
        return m_traversable_graph_flags;
    }

    inline unsigned int depth() const 
    {
        return m_depth;
    }

    inline unsigned int requiredSBTEntries() const 
    {
        return m_required_sbt_entries;
    }

    OptixSceneSBT sbt_data;

    void addEventReceiver(OptixSceneEventReceiverPtr rec);
    void removeEventReceiver(OptixSceneEventReceiverPtr rec);

private:
    OptixSceneCommitResult buildGAS();

    OptixSceneCommitResult buildIAS();

    void notifyEventReceivers(const OptixSceneCommitResult& info);
    

    OptixAccelerationStructurePtr m_as;

    OptixSceneType m_type = OptixSceneType::NONE;
    OptixGeometryType m_geom_type = OptixGeometryType::MESH;

    IDGen gen;

    std::map<unsigned int, OptixGeometryPtr> m_geometries;
    std::unordered_map<OptixGeometryPtr, unsigned int> m_ids;

    std::unordered_set<OptixInstWPtr> m_parents;

    std::unordered_set<OptixSceneEventReceiverWPtr> m_event_rec;

    bool m_geom_added = false;
    bool m_geom_removed = false;

    // filled after commit
    unsigned int m_traversable_graph_flags = 0;
    unsigned int m_depth = 0;
    unsigned int m_required_sbt_entries = 0;
};

OptixScenePtr make_optix_scene(
    const aiScene* ascene, 
    OptixContextPtr context = optix_default_context());

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SCENE_HPP