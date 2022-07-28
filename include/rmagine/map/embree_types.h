#ifndef RMAGINE_MAP_EMBREE_TYPES_H
#define RMAGINE_MAP_EMBREE_TYPES_H


#include <memory>
#include <unordered_set>
#include <set>
#include <functional>

namespace rmagine 
{

class EmbreeDevice;
class EmbreeGeometry;
class EmbreeMesh;
class EmbreeInstance;
class EmbreeScene;
class EmbreeMap;

using EmbreeDevicePtr = std::shared_ptr<EmbreeDevice>; 
using EmbreeGeometryPtr = std::shared_ptr<EmbreeGeometry>;
using EmbreeMeshPtr = std::shared_ptr<EmbreeMesh>;
using EmbreeInstancePtr = std::shared_ptr<EmbreeInstance>;
using EmbreeScenePtr = std::shared_ptr<EmbreeScene>;
using EmbreeMapPtr = std::shared_ptr<EmbreeMap>;



using EmbreeDeviceWPtr = std::weak_ptr<EmbreeDevice>; 
using EmbreeGeometryWPtr = std::weak_ptr<EmbreeGeometry>;
using EmbreeMeshWPtr = std::weak_ptr<EmbreeMesh>;
using EmbreeInstanceWPtr = std::weak_ptr<EmbreeInstance>;
using EmbreeSceneWPtr = std::weak_ptr<EmbreeScene>;
using EmbreeMapWPtr = std::weak_ptr<EmbreeMap>;


using EmbreeInstanceSet = std::unordered_set<EmbreeInstancePtr>;
using EmbreeMeshSet = std::unordered_set<EmbreeMeshPtr>;
using EmbreeGeometrySet = std::unordered_set<EmbreeGeometryPtr>;


template<typename T>
struct lex_compare {
    bool operator() (const std::weak_ptr<T> &lhs, const std::weak_ptr<T> &rhs) const 
    {
        auto lptr = lhs.lock(), rptr = rhs.lock();
        if (!rptr) return false; // nothing after expired pointer 
        if (!lptr) return true;  // every not expired after expired pointer
        return lptr != rptr;
    }
};

using EmbreeGeometryWSet = std::set<EmbreeGeometryWPtr, lex_compare<EmbreeGeometry> >;
using EmbreeInstanceWSet = std::set<EmbreeInstanceWPtr, lex_compare<EmbreeInstance> >;
using EmbreeMeshWSet = std::set<EmbreeMeshWPtr, lex_compare<EmbreeInstance> >;

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_TYPES_H