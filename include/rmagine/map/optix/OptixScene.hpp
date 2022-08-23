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


#include <rmagine/simulation/optix/OptixProgramMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

#include <rmagine/util/optix/OptixSbtRecord.hpp>


namespace rmagine
{

struct OptixSensorProgram
{
    OptixSensorPipelinePtr pipeline;
    OptixSBTPtr            sbt;
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
    void commit();

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

    // 
    OptixSensorProgram registerSensorProgram(const OptixSimulationDataGeneric& flags);



private:
    void buildGAS();

    void buildIAS();

    void updateSBT();

    OptixAccelerationStructurePtr m_as;

    OptixSceneType m_type = OptixSceneType::NONE;
    OptixGeometryType m_geom_type = OptixGeometryType::MESH;

    IDGen gen;

    std::map<unsigned int, OptixGeometryPtr> m_geometries;
    std::unordered_map<OptixGeometryPtr, unsigned int> m_ids;

    std::unordered_set<OptixInstWPtr> m_parents;

    bool m_geom_added = false;
    bool m_geom_removed = false;

    // filled after commit
    unsigned int m_traversable_graph_flags = 0;
    unsigned int m_depth = 0;
    unsigned int m_required_sbt_entries = 0;




    // filled after commit and first sensor usage

    using RayGenData        = RayGenDataEmpty;
    using MissData          = MissDataEmpty;
    using HitGroupData      = OptixSceneSBT;

    using RayGenSbtRecord   = SbtRecord<RayGenData>;
    using MissSbtRecord     = SbtRecord<MissData>;
    using HitGroupSbtRecord = SbtRecord<HitGroupData>;


    OptixPipelineCompileOptions m_pipeline_compile_options;
    OptixProgramGroupOptions m_program_group_options;
    

    // sensor model type id -> RayGenModule
    std::unordered_map<unsigned int, RayGenModulePtr>  m_sensor_raygen_modules;
    // bounding bools key -> hit module
    std::unordered_map<unsigned int, HitModulePtr>     m_hit_modules;

    std::unordered_map<OptixSimulationDataGeneric, OptixSensorPipelinePtr> m_pipelines;
    // bounding bools key -> sbt
    std::unordered_map<OptixSimulationDataGeneric, OptixSBTPtr> m_sbts;




    const unsigned int m_semantics[8] = {
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ
    };

    OptixPayloadType m_payload_type;
};

OptixScenePtr make_optix_scene(const aiScene* ascene, OptixContextPtr context = optix_default_context());

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SCENE_HPP