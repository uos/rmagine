#ifndef RMAGINE_MAP_EMBREE_INSTANCE_HPP
#define RMAGINE_MAP_EMBREE_INSTANCE_HPP

#include "embree_types.h"

#include <rmagine/math/types.h>
#include <rmagine/types/mesh_types.h>
#include <unordered_set>
#include <memory>

#include <embree3/rtcore.h>

#include <functional>

namespace rmagine
{

/**
 * @brief EmbreeInstance
 * 
 * N instances belongs to 1 scene
 * M instances of 1 mesh
 * 
 * N instance belongs to 1 parent scene
 * 1 instance has one child scene
 * 
 */
class EmbreeInstance
: public std::enable_shared_from_this<EmbreeInstance>
{
public:
    EmbreeInstance(EmbreeDevicePtr device);
    ~EmbreeInstance();

    Matrix4x4 T;

    void setTransform(const Matrix4x4& T);
    void setTransform(const Transform& T);

    // embree fields
    RTCGeometry handle() const;

    void set(EmbreeScenePtr scene);
    EmbreeScenePtr scene();

    // Make this more comfortable to use
    // - functions as: setMesh(), or addMesh() ?
    // - translate rotate scale? 

    /**
     * @brief Call update after changing the transformation. TODO TEST
     * 
     */
    void commit();

    void disable();

    void release();

    EmbreeScenePtr parent;
    // id only valid if parent is set
    unsigned int id;
private:
    RTCGeometry m_handle;
    EmbreeScenePtr m_scene;

    EmbreeDevicePtr m_device;
};




} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_INSTANCE_HPP