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
    EmbreeMesh( EmbreeDevicePtr device);

    EmbreeMesh( EmbreeDevicePtr device, 
                unsigned int Nvertices, 
                unsigned int Nfaces);

    EmbreeMesh( EmbreeDevicePtr device,
                const aiMesh* amesh);

    ~EmbreeMesh();

    Memory<Vector, RAM> vertices;

    unsigned int Nfaces;
    Face* faces;
    
    // more custom attributes
    Memory<Vector, RAM> normals;

    RTCGeometry handle() const;

    void setScale(const Vector3& S);
    void setTransform(const Matrix4x4& T);
    void setTransform(const Transform& T);

    Transform transform() const;
    Vector3 scale() const;

    void apply();

    void commit();
    void release();
    void disable();
    void enable();

    void markAsChanged();

    // embree fields
    EmbreeScenePtr parent;
    // id only valid if parent is set
    unsigned int id;
private:

    // embree constructed buffers
    unsigned int Nvertices;
    Vertex* vertices_transformed;

    Memory<Vector, RAM> normals_transformed;

    Transform m_T;
    Vector3 m_S;

    RTCGeometry m_handle;

    // connections
    EmbreeDevicePtr m_device;
};

using EmbreeMeshPtr = std::shared_ptr<EmbreeMesh>;

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_MESH_HPP