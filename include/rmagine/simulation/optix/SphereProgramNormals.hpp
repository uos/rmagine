#ifndef RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_NORMALS_HPP
#define RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_NORMALS_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

#include "rmagine/util/optix/OptixSbtRecord.hpp"
#include "rmagine/util/optix/OptixData.hpp"

namespace rmagine {

typedef SbtRecord<HitGroupDataNormals>   HitGroupSbtRecord;

class SphereProgramNormals : public OptixProgram
{
public:
    SphereProgramNormals(OptixMapPtr map);
    ~SphereProgramNormals();

private:
    HitGroupSbtRecord m_hg_sbt;
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_NORMALS_HPP