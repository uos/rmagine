#include "rmagine/map/EmbreeMap.hpp"

// other internal deps
#include "rmagine/map/embree/EmbreeDevice.hpp"
#include "rmagine/map/embree/EmbreeScene.hpp"

#include <iostream>

#include <map>
#include <cassert>

namespace rmagine {

EmbreeMap::EmbreeMap(EmbreeDevicePtr device)
:device(device)
{
    rtcInitPointQueryContext(&pq_context);
}

EmbreeMap::EmbreeMap(EmbreeScenePtr scene)
:EmbreeMap(scene->device())
{
    this->scene = scene;
}

EmbreeMap::~EmbreeMap()
{
    
}

Point EmbreeMap::closestPoint(const Point& qp)
{
    // std::cout << "TODO23: check if closestPoint is working after refactoring" << std::endl;
    RTCPointQuery query;
    query.x = qp.x; 
    query.y = qp.y;
    query.z = qp.z;
    query.radius = std::numeric_limits<float>::max();
    query.time = 0.0;

    ClosestPointResult result;

    PointQueryUserData user_data;
    user_data.scene = &scene;
    user_data.result = &result;
    rtcPointQuery(scene->handle(), &query, &pq_context, nullptr, (void*)&user_data);

    if(result.geomID == RTC_INVALID_GEOMETRY_ID || result.primID == RTC_INVALID_GEOMETRY_ID)
    {
        std::cout << "RESULT SEEMS TO BE WRONG" << std::endl;
        // throw std::runtime_error("Cannot find nearest point on surface");
    }

    return result.p;
}

} // namespace rmagine