#ifndef RMAGINE_MAP_OPTIX_DEFINTIONS_HPP
#define RMAGINE_MAP_OPTIX_DEFINTIONS_HPP

#include <memory>
#include <rmagine/util/hashing.h>

namespace rmagine
{

class OptixEntity;
class OptixTransformable;
class OptixAccelerationStructure;
class OptixGeometry;
class OptixInst;
class OptixInstances;
class OptixScene;

using OptixEntityPtr = std::shared_ptr<OptixEntity>;
using OptixTransformablePtr = std::shared_ptr<OptixTransformable>;
using OptixAccelerationStructurePtr = std::shared_ptr<OptixAccelerationStructure>;
using OptixGeometryPtr = std::shared_ptr<OptixGeometry>;
using OptixInstPtr = std::shared_ptr<OptixInst>;
using OptixInstancesPtr = std::shared_ptr<OptixInstances>;
using OptixScenePtr = std::shared_ptr<OptixScene>;

using OptixEntityWPtr = std::weak_ptr<OptixEntity>;
using OptixTransformableWPtr = std::weak_ptr<OptixTransformable>;
using OptixAccelerationStructureWPtr = std::weak_ptr<OptixAccelerationStructure>;
using OptixGeometryWPtr = std::weak_ptr<OptixGeometry>;
using OptixInstWPtr = std::weak_ptr<OptixInst>;
using OptixInstancesWPtr = std::weak_ptr<OptixInstances>;
using OptixSceneWPtr = std::weak_ptr<OptixScene>;




} // namespace rmagine

namespace std
{

// INSTANCE
template<>
struct hash<rmagine::OptixInstWPtr> 
    : public rmagine::weak_hash<rmagine::OptixInst>
{};

template<>
struct equal_to<rmagine::OptixInstWPtr> 
    : public rmagine::weak_equal_to<rmagine::OptixInst>
{};

template<>
struct less<rmagine::OptixInstWPtr> 
    : public rmagine::weak_less<rmagine::OptixInst>
{};

} // namespace std

#endif // RMAGINE_MAP_OPTIX_DEFINTIONS_HPP