#ifndef RMAGINE_SIMULATION_OPTIX_PROGRAM_MAP_HPP
#define RMAGINE_SIMULATION_OPTIX_PROGRAM_MAP_HPP

#include <unordered_map>
#include <rmagine/simulation/optix/OptixSimulationData.hpp>

namespace std {

template <typename ModelT>
struct hash<rmagine::OptixSimulationDataGeneric_<ModelT> >
{
    std::size_t operator()(const rmagine::OptixSimulationDataGeneric_<ModelT>& k) const
    {
        unsigned int hitsKey = static_cast<unsigned int>(k.computeHits) << 0;
        unsigned int rangesKey = static_cast<unsigned int>(k.computeRanges) << 1;
        unsigned int pointKey = static_cast<unsigned int>(k.computePoints) << 2;
        unsigned int normalsKey = static_cast<unsigned int>(k.computeNormals) << 3;
        unsigned int faceIdsKey = static_cast<unsigned int>(k.computeFaceIds) << 4;
        unsigned int geomIdsKey = static_cast<unsigned int>(k.computeGeomIds) << 5;
        unsigned int objectIdsKey = static_cast<unsigned int>(k.computeObjectIds) << 6;
        // bitwise or
        return (hitsKey | rangesKey | pointKey | normalsKey | faceIdsKey | geomIdsKey | objectIdsKey);
    }
};

template<>
struct hash<rmagine::OptixSimulationDataGeneric >
{
    std::size_t operator()(const rmagine::OptixSimulationDataGeneric& k) const
    {
        // first 24 bits are reserved for bool flags (bound values)
        std::size_t hitsKey = static_cast<std::size_t>(k.computeHits) << 0;
        std::size_t rangesKey = static_cast<std::size_t>(k.computeRanges) << 1;
        std::size_t pointKey = static_cast<std::size_t>(k.computePoints) << 2;
        std::size_t normalsKey = static_cast<std::size_t>(k.computeNormals) << 3;
        std::size_t faceIdsKey = static_cast<std::size_t>(k.computeFaceIds) << 4;
        std::size_t geomIdsKey = static_cast<std::size_t>(k.computeGeomIds) << 5;
        std::size_t objectIdsKey = static_cast<std::size_t>(k.computeObjectIds) << 6;

        // next 8 bit are reserved for sensor type
        // sensor_type should not be higher than 2**8=256
        std::size_t sensorTypeKey = static_cast<std::size_t>(k.model_type) << 24;
        
        // bitwise or
        return (hitsKey | rangesKey | pointKey | normalsKey | faceIdsKey | geomIdsKey | objectIdsKey);
    }
};

} // namespace std

namespace rmagine
{

template<typename ModelT>
bool operator==(const OptixSimulationDataGeneric_<ModelT> &a, const OptixSimulationDataGeneric_<ModelT> &b)
{ 
    return std::hash<OptixSimulationDataGeneric_<ModelT> >()(a) == std::hash<OptixSimulationDataGeneric_<ModelT> >()(b);
}

inline bool operator==(const OptixSimulationDataGeneric &a, const OptixSimulationDataGeneric &b)
{ 
    return std::hash<OptixSimulationDataGeneric>()(a) == std::hash<OptixSimulationDataGeneric>()(b);
}

} // namespace rmagine

#endif // RMAGINE_SIMULATION_OPTIX_PROGRAM_MAP_HPP