#ifndef IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_NORMALS_HPP
#define IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_NORMALS_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>

namespace imagine {

class PinholeProgramNormals : public OptixProgram
{
public:
    PinholeProgramNormals(OptixMapPtr map);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_NORMALS_HPP