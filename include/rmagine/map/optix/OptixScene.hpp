#ifndef RMAGINE_MAP_OPTIX_SCENE_HPP
#define RMAGINE_MAP_OPTIX_SCENE_HPP

#include <rmagine/util/optix/OptixContext.hpp>
#include <rmagine/util/IDGen.hpp>

#include <optix.h>

#include "optix_definitions.h"

#include "OptixEntity.hpp"

#include <map>
#include <rmagine/util/IDGen.hpp>

#include <assimp/scene.h>

#include <rmagine/util/optix/OptixData.hpp>
#include <rmagine/types/MemoryCuda.hpp>

#include "rmagine/util/optix/OptixSbtRecord.hpp"

namespace rmagine
{

// typedef SbtRecord<HitGroupDataScene>      HitGroupSbtRecordScene;

class OptixScene 
: public OptixEntity
{
public:
    OptixScene(OptixContextPtr context = optix_default_context());
    OptixScene(OptixGeometryPtr geom, OptixContextPtr context = optix_default_context());

    virtual ~OptixScene();

    void setRoot(OptixGeometryPtr root);
    OptixGeometryPtr getRoot() const;

    unsigned int add(OptixGeometryPtr geom);
    unsigned int get(OptixGeometryPtr geom) const;
    std::map<unsigned int, OptixGeometryPtr> geometries() const;
    std::unordered_map<OptixGeometryPtr, unsigned int> ids() const;
    
    void commit();

    Memory<HitGroupDataScene, RAM> m_h_hitgroup_data;
    // Memory<HitGroupDataScene, VRAM_CUDA> m_hitgroup_data;

private:
    OptixGeometryPtr m_root;
    
    IDGen gen;

    std::map<unsigned int, OptixGeometryPtr> m_geometries;
    std::unordered_map<OptixGeometryPtr, unsigned int> m_ids;
};

OptixScenePtr make_optix_scene(const aiScene* ascene, OptixContextPtr context = optix_default_context());

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SCENE_HPP