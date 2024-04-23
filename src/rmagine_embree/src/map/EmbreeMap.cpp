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
    // rtcInitPointQueryContext(&pq_context);
}

EmbreeMap::EmbreeMap(EmbreeScenePtr scene)
:EmbreeMap(scene->device())
{
    this->scene = scene;
}

EmbreeMap::~EmbreeMap()
{
    
}

EmbreeClosestPointResult EmbreeMap::closestPoint(
  const Point& qp,
  const float& max_distance)
{
    return this->scene->closestPoint(qp, max_distance);
}

} // namespace rmagine