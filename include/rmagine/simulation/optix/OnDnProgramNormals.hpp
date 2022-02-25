#ifndef IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_NORMALS_HPP
#define IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_NORMALS_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

namespace rmagine {

class OnDnProgramNormals : public OptixProgram
{
public:
    OnDnProgramNormals(OptixMapPtr map);
};

} // namespace rmagine

#endif // IMAGINE_SIMULATION_OPTIX_ONDN_PROGRAM_NORMALS_HPP