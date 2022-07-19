#ifndef RMAGINE_MAP_EMBREE_TYPES_H
#define RMAGINE_MAP_EMBREE_TYPES_H


#include <memory>
#include <unordered_set>
#include <functional>

namespace rmagine 
{

class EmbreeDevice;
class EmbreeMesh;
class EmbreeInstance;
class EmbreeScene;
class EmbreeMap;

using EmbreeDevicePtr = std::shared_ptr<EmbreeDevice>; 
using EmbreeMeshPtr = std::shared_ptr<EmbreeMesh>;
using EmbreeInstancePtr = std::shared_ptr<EmbreeInstance>;
using EmbreeScenePtr = std::shared_ptr<EmbreeScene>;
using EmbreeMapPtr = std::shared_ptr<EmbreeMap>;


using EmbreeInstanceSet = std::unordered_set<EmbreeInstancePtr>;
using EmbreeMeshSet = std::unordered_set<EmbreeMeshPtr>;



} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_TYPES_H