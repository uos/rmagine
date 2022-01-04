#ifndef IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_NORMALS_HPP
#define IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_NORMALS_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>

namespace imagine {

class OnDnProgramNormals : public OptixProgram
{
public:
    OnDnProgramNormals(OptixMapPtr map);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_NORMALS_HPP