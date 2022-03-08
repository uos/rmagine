#ifndef RMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_NORMALS_HPP
#define RMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_NORMALS_HPP

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>

namespace rmagine {

class PinholeProgramNormals : public OptixProgram
{
public:
    PinholeProgramNormals(OptixMapPtr map);
};

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_PINHOLE_PROGRAM_NORMALS_HPP