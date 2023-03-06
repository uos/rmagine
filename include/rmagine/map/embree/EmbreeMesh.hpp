/*
 * Copyright (c) 2022, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * 
 * @brief EmbreeMesh
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_EMBREE_MESH_HPP
#define RMAGINE_MAP_EMBREE_MESH_HPP

#include "embree_definitions.h"

#include <rmagine/types/Memory.hpp>
#include <assimp/mesh.h>

#include <rmagine/math/types.h>
#include <rmagine/types/mesh_types.h>

#include <memory>

#if RMAGINE_EMBREE_VERSION_MAJOR == 3
#include <embree3/rtcore.h>
#elif RMAGINE_EMBREE_VERSION_MAJOR == 4
#include <embree4/rtcore.h>
#else
#pragma message("Wrong major version of Embree found: ", RMAGINE_EMBREE_VERSION_MAJOR)
#endif

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

    void initVertexNormals();

    // PUBLIC ATTRIBUTES
    MemoryView<Face, RAM> faces() const;
    MemoryView<Vertex, RAM> vertices() const;
    MemoryView<Vector, RAM> vertexNormals() const;
    MemoryView<Vector, RAM> faceNormals() const;
    
    MemoryView<const Vertex, RAM> verticesTransformed() const;
    
    
    

    void computeFaceNormals();

    /**
     * @brief Apply new Transform and Scale to buffers
     * 
     */
    void apply();

    virtual EmbreeGeometryType type() const
    {
        return EmbreeGeometryType::MESH;
    }


    // embree constructed buffers
protected:
    unsigned int m_num_vertices = 0;
    unsigned int m_num_faces = 0;

    Memory<Vertex, RAM> m_vertices;
    Face* m_faces;
    // vertex and face normals
    Memory<Vector, RAM> m_vertex_normals;
    Memory<Vector, RAM> m_face_normals;

private:

    // after transform
    Vertex* m_vertices_transformed;
    Memory<Vector, RAM> m_face_normals_transformed;
    Memory<Vector, RAM> m_vertex_normals_transformed;
};

using EmbreeMeshPtr = std::shared_ptr<EmbreeMesh>;

} // namespace rmagine

#endif // RMAGINE_MAP_EMBREE_MESH_HPP