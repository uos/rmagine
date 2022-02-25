#ifndef IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_NORMALS_HPP
#define IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_NORMALS_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

namespace rmagine {

class PinholeProgramNormals : public OptixProgram
{
public:
    PinholeProgramNormals(OptixMapPtr map);
};

} // namespace rmagine

#endif // IMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_NORMALS_HPP