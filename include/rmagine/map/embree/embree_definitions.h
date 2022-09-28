#ifndef RMAGINE_MAP_EMBREE_TYPES_H
#define RMAGINE_MAP_EMBREE_TYPES_H


#include <memory>
#include <unordered_set>
#include <set>
#include <functional>
#include <iostream>
#include <utility>

#include <rmagine/util/hashing.h>

namespace rmagine 
{

enum class EmbreeGeometryType 
{
    INSTANCE,
    MESH,
    POINTS
};

class EmbreeDevice;
class EmbreeGeometry;
class EmbreeMesh;
class EmbreeInstance;
class EmbreeScene;

using EmbreeDevicePtr = std::shared_ptr<EmbreeDevice>; 
using EmbreeGeometryPtr = std::shared_ptr<EmbreeGeometry>;
using EmbreeMeshPtr = std::shared_ptr<EmbreeMesh>;
using EmbreeInstancePtr = std::shared_ptr<EmbreeInstance>;
using EmbreeScenePtr = std::shared_ptr<EmbreeScene>;

using EmbreeDeviceConstPtr = std::shared_ptr<const EmbreeDevice>; 
using EmbreeGeometryConstPtr = std::shared_ptr<const EmbreeGeometry>;
using EmbreeMeshConstPtr = std::shared_ptr<const EmbreeMesh>;
using EmbreeInstanceConstPtr = std::shared_ptr<const EmbreeInstance>;
using EmbreeSceneConstPtr = std::shared_ptr<const EmbreeScene>;

using EmbreeDeviceWPtr = std::weak_ptr<EmbreeDevice>; 
using EmbreeGeometryWPtr = std::weak_ptr<EmbreeGeometry>;
using EmbreeMeshWPtr = std::weak_ptr<EmbreeMesh>;
using EmbreeInstanceWPtr = std::weak_ptr<EmbreeInstance>;
using EmbreeSceneWPtr = std::weak_ptr<EmbreeScene>;

using EmbreeDeviceWConstPtr = std::weak_ptr<const EmbreeDevice>; 
using EmbreeGeometryWConstPtr = std::weak_ptr<const EmbreeGeometry>;
using EmbreeMeshWConstPtr = std::weak_ptr<const EmbreeMesh>;
using EmbreeInstanceWConstPtr = std::weak_ptr<const EmbreeInstance>;
using EmbreeSceneWConstPtr = std::weak_ptr<const EmbreeScene>;


using EmbreeInstanceSet = std::unordered_set<EmbreeInstancePtr>;
using EmbreeMeshSet = std::unordered_set<EmbreeMeshPtr>;
using EmbreeGeometrySet = std::unordered_set<EmbreeGeometryPtr>;




using EmbreeGeometryWSet = std::set<EmbreeGeometryWPtr, weak_less<EmbreeGeometry> >;
using EmbreeInstanceWSet = std::set<EmbreeInstanceWPtr, weak_less<EmbreeInstance> >;
using EmbreeMeshWSet = std::set<EmbreeMeshWPtr, weak_less<EmbreeMesh> >;
using EmbreeSceneWSet = std::set<EmbreeSceneWPtr, weak_less<EmbreeScene> >;


template<typename T>
using WeakUnorderedSet = std::unordered_set<std::weak_ptr<T>, weak_hash<T>, weak_equal_to<T> >;


} // namespace rmagine

namespace std
{

// need to define std::hash, std::equal_to, std::less to shorter write 
// std::set<TypeWPtr>, std::unordered_set<TypeWPtr> ...


/// GEOMETRY

template<>
struct hash<rmagine::EmbreeGeometryWPtr> 
    : public rmagine::weak_hash<rmagine::EmbreeGeometry>
{};

template<>
struct equal_to<rmagine::EmbreeGeometryWPtr> 
    : public rmagine::weak_equal_to<rmagine::EmbreeGeometry>
{};

template<>
struct less<rmagine::EmbreeGeometryWPtr> 
    : public rmagine::weak_less<rmagine::EmbreeGeometry>
{};

/// SCENE
template<>
struct hash<rmagine::EmbreeSceneWPtr> 
    : public rmagine::weak_hash<rmagine::EmbreeScene>
{};

template<>
struct equal_to<rmagine::EmbreeSceneWPtr> 
    : public rmagine::weak_equal_to<rmagine::EmbreeScene>
{};

template<>
struct less<rmagine::EmbreeSceneWPtr> 
    : public rmagine::weak_less<rmagine::EmbreeScene>
{};

/// INSTANCE
template<>
struct hash<rmagine::EmbreeInstanceWPtr> 
    : public rmagine::weak_hash<rmagine::EmbreeInstance>
{};

template<>
struct equal_to<rmagine::EmbreeInstanceWPtr> 
    : public rmagine::weak_equal_to<rmagine::EmbreeInstance>
{};

template<>
struct less<rmagine::EmbreeInstanceWPtr> 
    : public rmagine::weak_less<rmagine::EmbreeInstance>
{};

/// MESH
template<>
struct hash<rmagine::EmbreeMeshWPtr> 
    : public rmagine::weak_hash<rmagine::EmbreeMesh>
{};

template<>
struct equal_to<rmagine::EmbreeMeshWPtr> 
    : public rmagine::weak_equal_to<rmagine::EmbreeMesh>
{};

template<>
struct less<rmagine::EmbreeMeshWPtr> 
    : public rmagine::weak_less<rmagine::EmbreeMesh>
{};


} // namespace std

#endif // RMAGINE_MAP_EMBREE_TYPES_H