#ifndef IMAGINE_SIMULATION_OPTIX_SCAN_PROGRAM_RANGE_HPP
#define IMAGINE_SIMULATION_OPTIX_SCAN_PROGRAM_RANGE_HPP

#include <imagine/map/OptixMap.hpp>
#include <imagine/util/optix/OptixProgram.hpp>

namespace imagine {

class ScanProgramRanges : public OptixProgram
{
public:
    ScanProgramRanges(OptixMapPtr map);
};

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_SCAN_PROGRAM_RANGE_HPP