#ifndef IMAGINE_SIMULATION_OPTIX_PROGRAM_MAP_HPP
#define IMAGINE_SIMULATION_OPTIX_PROGRAM_MAP_HPP

#include <unordered_map>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>
#include <rmagine/util/optix/OptixProgram.hpp>


namespace std {

template <typename ModelT>
struct hash<rmagine::OptixSimulationDataGeneric<ModelT> >
{
    std::size_t operator()(const rmagine::OptixSimulationDataGeneric<ModelT>& k) const
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

namespace rmagine
{

template<typename ModelT>
bool operator==(const OptixSimulationDataGeneric<ModelT> &a, const OptixSimulationDataGeneric<ModelT> &b)
{ 
    return (a.computeHits == b.computeHits
            && a.computeRanges == b.computeRanges
            && a.computePoints == b.computePoints
            && a.computeNormals == b.computeNormals
            && a.computeFaceIds == b.computeFaceIds
            && a.computeObjectIds == b.computeObjectIds );
}

} // namespace rmagine

#endif // IMAGINE_SIMULATION_OPTIX_PROGRAM_MAP_HPP