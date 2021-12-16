#ifndef IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_NORMALS_HPP
#define IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_NORMALS_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>

namespace imagine {

class O1DnProgramNormals : public OptixProgram
{
public:
    O1DnProgramNormals(OptixMapPtr map);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_NORMALS_HPP