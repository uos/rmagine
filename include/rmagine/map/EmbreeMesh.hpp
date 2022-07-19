#ifndef RMAGINE_MAP_EMBREE_MESH_HPP
#define RMAGINE_MAP_EMBREE_MESH_HPP

#include "embree_types.h"

#include <rmagine/types/Memory.hpp>
#include <assimp/mesh.h>

#include <rmagine/math/types.h>
#include <rmagine/types/mesh_types.h>

#include <memory>
#include <embree3/rtcore.h>


namespace rmagine
{

/**
 * @brief EmbreeMesh
 * 
 * N instances of 1 mesh
 * -> mesh can attached multiple times to scene
 * mesh can also exist without instance
 * -> mesh can attached to scene only once
 * 
 */
class EmbreeMesh
{
public:
    // TODO: constructor destructor

    EmbreeMesh( EmbreeDevicePtr device);

    EmbreeMesh( EmbreeDevicePtr device, 
                unsigned int Nvertices, 
                unsigned int Nfaces);

    EmbreeMesh( EmbreeDevicePtr device,
                const aiMesh* amesh);

    // embree constructed buffers
    unsigned int Nvertices;
    Vertex* vertices;

    unsigned int Nfaces;
    Face* faces;
    
    // more custom attributes
    Memory<Vector, RAM> normals;

    // embree fields
    RTCGeometry handle;
    unsigned int geomID;

    void transform(const Matrix4x4& T);
    
    void setScene(EmbreeScenePtr scene);
    void setNewScene();
    EmbreeScenePtr scene();

    void addInstance(EmbreeInstancePtr instance);
    bool hasInstance(EmbreeInstancePtr instance) const;
    EmbreeInstanceSet instances();

    void commit();
private:

    // connections
    EmbreeInstanceSet m_instances;
    EmbreeScenePtr m_scene;
    EmbreeDevicePtr m_device;
};

using EmbreeMeshPtr = std::shared_ptr<EmbreeMesh>;

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_MESH_HPP