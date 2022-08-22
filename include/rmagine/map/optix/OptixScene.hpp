#ifndef RMAGINE_MAP_OPTIX_SCENE_HPP
#define RMAGINE_MAP_OPTIX_SCENE_HPP

#include <rmagine/util/optix/OptixContext.hpp>
#include <rmagine/util/IDGen.hpp>

#include <optix.h>

#include "optix_definitions.h"
#include "optix_sbt.h"

#include "OptixEntity.hpp"

#include <map>
#include <rmagine/util/IDGen.hpp>

#include <assimp/scene.h>

#include <rmagine/types/MemoryCuda.hpp>
#include <unordered_set>

namespace rmagine
{

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
    
    void commit();
    unsigned int depth() const;

    OptixInstPtr instantiate();

    inline OptixAccelerationStructurePtr as() const
    {
        return m_as;
    }

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

    OptixSceneSBT sbt_data;

    unsigned int required_sbt_entries = 0;

private:
    void buildGAS();

    void buildIAS();

    OptixAccelerationStructurePtr m_as;

    OptixSceneType m_type = OptixSceneType::NONE;
    OptixGeometryType m_geom_type = OptixGeometryType::MESH;

    IDGen gen;

    std::map<unsigned int, OptixGeometryPtr> m_geometries;
    std::unordered_map<OptixGeometryPtr, unsigned int> m_ids;

    std::unordered_set<OptixInstWPtr> m_parents;

    bool m_geom_added = false;
    bool m_geom_removed = false;
};

OptixScenePtr make_optix_scene(const aiScene* ascene, OptixContextPtr context = optix_default_context());

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SCENE_HPP