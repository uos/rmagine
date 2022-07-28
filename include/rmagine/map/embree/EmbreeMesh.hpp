#ifndef RMAGINE_MAP_EMBREE_MESH_HPP
#define RMAGINE_MAP_EMBREE_MESH_HPP

#include "embree_types.h"

#include <rmagine/types/Memory.hpp>
#include <assimp/mesh.h>

#include <rmagine/math/types.h>
#include <rmagine/types/mesh_types.h>

#include <memory>
#include <embree3/rtcore.h>
#include "EmbreeDevice.hpp"
#include "EmbreeGeometry.hpp"

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
: public EmbreeGeometry
{
public:
    using Base = EmbreeGeometry;
    EmbreeMesh( EmbreeDevicePtr device = embree_default_device());

    EmbreeMesh( unsigned int Nvertices, 
                unsigned int Nfaces,
                EmbreeDevicePtr device = embree_default_device());

    EmbreeMesh( const aiMesh* amesh,
                EmbreeDevicePtr device = embree_default_device());

    virtual ~EmbreeMesh();

    void init(unsigned int Nvertices, unsigned int Nfaces);
    void init(const aiMesh* amesh);

    // PUBLIC ATTRIBUTES
    Memory<Vector, RAM> vertices;

    unsigned int Nfaces;
    Face* faces;
    
    // vertex and face normals
    Memory<Vector, RAM> vertex_normals;
    Memory<Vector, RAM> face_normals;


    /**
     * @brief Apply new Transform and Scale to buffers
     * 
     */
    void apply();

    void computeFaceNormals();

    void markAsChanged();
private:
    // embree constructed buffers
    unsigned int Nvertices;
    Vertex* vertices_transformed;

    Memory<Vector, RAM> face_normals_transformed;
    Memory<Vector, RAM> vertex_normals_transformed;
};

using EmbreeMeshPtr = std::shared_ptr<EmbreeMesh>;

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_MESH_HPP