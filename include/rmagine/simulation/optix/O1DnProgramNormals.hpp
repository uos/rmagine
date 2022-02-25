#ifndef IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_NORMALS_HPP
#define IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_NORMALS_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

namespace rmagine {

class O1DnProgramNormals : public OptixProgram
{
public:
    O1DnProgramNormals(OptixMapPtr map);
};

} // namespace rmagine

#endif // IMAGINE_SIMULATION_OPTIX_O1DN_PROGRAM_NORMALS_HPP