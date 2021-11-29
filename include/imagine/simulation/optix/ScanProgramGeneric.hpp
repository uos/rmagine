#ifndef IMAGINE_SIMULATION_OPTIX_SCAN_PROGRAM_GENERIC_HPP
#define IMAGINE_SIMULATION_OPTIX_SCAN_PROGRAM_GENERIC_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>

namespace imagine {

class ScanProgramGeneric : public OptixProgram
{
public:
    ScanProgramGeneric(OptixMapPtr map);

    OptixProgramGroup miss_prog_groups[2];
    OptixProgramGroup hitgroup_prog_groups[2];
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_SCAN_PROGRAM_GENERIC_HPP