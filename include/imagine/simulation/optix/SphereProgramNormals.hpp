#ifndef IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_NORMALS_HPP
#define IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_NORMALS_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>

#include "imagine/util/optix/OptixSbtRecord.hpp"
#include "imagine/util/optix/OptixData.hpp"

namespace imagine {

typedef SbtRecord<HitGroupDataNormals>   HitGroupSbtRecord;

class SphereProgramNormals : public OptixProgram
{
public:
    SphereProgramNormals(OptixMapPtr map);
    ~SphereProgramNormals();

private:
    HitGroupSbtRecord m_hg_sbt;
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_SPHERE_PROGRAM_NORMALS_HPP