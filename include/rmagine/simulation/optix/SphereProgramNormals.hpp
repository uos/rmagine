#ifndef RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_NORMALS_HPP
#define RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_NORMALS_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

#include <rmagine/util/optix/OptixSbtRecord.hpp>
#include <rmagine/util/optix/OptixData.hpp>

namespace rmagine {

class SphereProgramNormals : public OptixProgram
{
    using RayGenData        = RayGenDataEmpty;
    using MissData          = MissDataEmpty;
    using HitGroupData      = HitGroupDataScene;

    using RayGenSbtRecord   = SbtRecord<RayGenData>;
    using MissSbtRecord     = SbtRecord<MissData>;
    using HitGroupSbtRecord = SbtRecord<HitGroupData>;

public:
    SphereProgramNormals(OptixMapPtr map);
    virtual ~SphereProgramNormals();

    void updateSBT();

private:
    // scene container
    OptixMapPtr         m_map;
    // currently used scene
    OptixScenePtr       m_scene;

    RayGenSbtRecord     rg_sbt;
    MissSbtRecord       ms_sbt;
    HitGroupSbtRecord   hg_sbt;
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_NORMALS_HPP