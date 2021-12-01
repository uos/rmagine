#ifndef IMAGINE_SIMULATION_OPTIX_PROGRAM_MAP_HPP
#define IMAGINE_SIMULATION_OPTIX_PROGRAM_MAP_HPP

#include <unordered_map>
#include <imagine/simulation/optix/OptixSimulationData.hpp>
#include <imagine/util/optix/OptixProgram.hpp>


namespace std {

template <>
struct hash<imagine::OptixSimulationDataGeneric>
{
    std::size_t operator()(const imagine::OptixSimulationDataGeneric& k) const
    {
        unsigned int hitsKey = static_cast<unsigned int>(k.computeHits) << 0;
        unsigned int rangesKey = static_cast<unsigned int>(k.computeRanges) << 1;
        unsigned int pointKey = static_cast<unsigned int>(k.computeRanges) << 2;
        unsigned int normalsKey = static_cast<unsigned int>(k.computeRanges) << 3;
        unsigned int faceIdsKey = static_cast<unsigned int>(k.computeRanges) << 4;
        unsigned int objectIdsKey = static_cast<unsigned int>(k.computeRanges) << 5;
        // bitwise or
        return (hitsKey | rangesKey | pointKey | normalsKey | faceIdsKey | objectIdsKey);
    }
};

} // namespace std

namespace imagine
{

bool operator==(const OptixSimulationDataGeneric &a, const OptixSimulationDataGeneric &b)
{ 
    return (a.computeHits == b.computeHits
            && a.computeRanges == b.computeRanges
            && a.computePoints == b.computePoints
            && a.computeNormals == b.computeNormals
            && a.computeFaceIds == b.computeFaceIds
            && a.computeObjectIds == b.computeObjectIds );
}

} // namespace imagine

#endif // IMAGINE_SIMULATION_OPTIX_PROGRAM_MAP_HPP