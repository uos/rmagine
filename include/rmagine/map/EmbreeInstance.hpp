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
 */
class EmbreeInstance
{
public:
    EmbreeInstance(EmbreeDevicePtr device);
    ~EmbreeInstance();

    Matrix4x4 T;

    // embree fields
    RTCGeometry handle;
    unsigned int instID;

    unsigned int id() const;

    void setScene(EmbreeScenePtr scene);
    EmbreeScenePtr scene();

    void setMesh(EmbreeMeshPtr mesh);
    EmbreeMeshPtr mesh();

    // Make this more comfortable to use
    // - functions as: setMesh(), or addMesh() ?
    // - translate rotate scale? 

    /**
     * @brief Call update after changing the transformation. TODO TEST
     * 
     */
    void commit();

private:
    EmbreeMeshPtr m_mesh;
    

    EmbreeScenePtr m_scene;
    EmbreeDevicePtr m_device;
};




} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_INSTANCE_HPP