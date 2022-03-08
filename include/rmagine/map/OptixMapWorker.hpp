#ifndef RMAGINE_MAP_OPTIXMAPWORKER_HPP
#define RMAGINE_MAP_OPTIXMAPWORKER_HPP

#include "MapWorker.hpp"
#include "OptixMap.hpp"

namespace rmagine 
{



class OptixMapWorker : public MapWorker
{
public:
    virtual void setMap(OptixMapPtr map);



private:
    OptixMapPtr m_map;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIXMAPWORKER_HPP