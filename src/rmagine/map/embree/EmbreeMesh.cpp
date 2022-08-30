#include "rmagine/map/embree/EmbreeMesh.hpp"

// other internal deps
#include "rmagine/map/embree/EmbreeDevice.hpp"
#include "rmagine/map/embree/EmbreeScene.hpp"

#include <iostream>

#include <map>
#include <cassert>



#include <rmagine/math/assimp_conversions.h>

namespace rmagine {

EmbreeMesh::EmbreeMesh(EmbreeDevicePtr device)
:Base(device)
{
    m_handle = rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_TRIANGLE);
}

EmbreeMesh::EmbreeMesh( 
    unsigned int Nvertices, 
    unsigned int Nfaces,
    EmbreeDevicePtr device)
:EmbreeMesh(device)
{
    init(Nvertices, Nfaces);
}

EmbreeMesh::EmbreeMesh( 
    const aiMesh* amesh,
    EmbreeDevicePtr device)
:EmbreeMesh(device)
{
    init(amesh);
}

EmbreeMesh::~EmbreeMesh()
{
    // std::cout << "[EmbreeMesh::~EmbreeMesh()] destroyed." << std::endl;
}

void EmbreeMesh::init(
    unsigned int Nvertices, 
    unsigned int Nfaces)
{
    m_num_vertices = Nvertices;
    m_num_faces = Nfaces;

    m_vertices.resize(Nvertices);

    m_vertices_transformed = reinterpret_cast<Vertex*>(rtcSetNewGeometryBuffer(m_handle,
                                                RTC_BUFFER_TYPE_VERTEX,
                                                0,
                                                RTC_FORMAT_FLOAT3,
                                                sizeof(Vertex),
                                                m_num_vertices));

    m_faces = reinterpret_cast<Face*>(rtcSetNewGeometryBuffer(m_handle,
                                                    RTC_BUFFER_TYPE_INDEX,
                                                    0,
                                                    RTC_FORMAT_UINT3,
                                                    sizeof(Face),
                                                    m_num_faces));
}

void EmbreeMesh::init(
    const aiMesh* amesh)
{
    init(amesh->mNumVertices, amesh->mNumFaces);

    name = amesh->mName.C_Str();

    if(!(amesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) )
    {
        std::cout << "[EmbreeMesh::init()] WARNING: aiMesh* has no triangles. prim type: " << amesh->mPrimitiveTypes << std::endl; 
    }

    const aiVector3D* ai_vertices = amesh->mVertices;
    const aiFace* ai_faces = amesh->mFaces;

    // copy mesh to embree buffers
    for(unsigned int i=0; i<m_num_vertices; i++)
    {
        m_vertices[i] = convert(ai_vertices[i]);
    }

    for(int i=0; i<m_num_faces; i++)
    {
        m_faces[i] = {ai_faces[i].mIndices[0], ai_faces[i].mIndices[1], ai_faces[i].mIndices[2]};
    }

    if(amesh->HasNormals())
    {
        m_vertex_normals.resize(m_num_vertices);
        m_vertex_normals_transformed.resize(m_num_vertices);
        for(size_t i=0; i<m_num_vertices; i++)
        {
            m_vertex_normals[i] = convert(amesh->mNormals[i]);
        }
    }

    computeFaceNormals();

    apply();
}

void EmbreeMesh::initVertexNormals()
{
    m_vertex_normals.resize(m_num_vertices);
    m_vertex_normals_transformed.resize(m_num_vertices);
}

MemoryView<Face, RAM> EmbreeMesh::faces() const
{
    return MemoryView<Face, RAM>(m_faces, m_num_faces);
}

MemoryView<Vertex, RAM> EmbreeMesh::vertices() const
{
    return m_vertices;
}

MemoryView<Vector, RAM> EmbreeMesh::vertexNormals() const
{
    return m_vertex_normals;
}

MemoryView<Vector, RAM> EmbreeMesh::faceNormals() const
{
    return m_face_normals;
}

MemoryView<const Vertex, RAM> EmbreeMesh::verticesTransformed() const
{
    return MemoryView<const Vertex, RAM>(m_vertices_transformed, m_num_vertices);
}

void EmbreeMesh::computeFaceNormals()
{
    if(m_face_normals.size() != m_num_faces)
    {
        m_face_normals.resize(m_num_faces);
    }

    if(m_face_normals_transformed.size() != m_num_faces)
    {
        m_face_normals_transformed.resize(m_num_faces);
    }
    
    for(size_t i=0; i<m_num_faces; i++)
    {
        const Vector v0 = m_vertices[m_faces[i].v0];
        const Vector v1 = m_vertices[m_faces[i].v1];
        const Vector v2 = m_vertices[m_faces[i].v2];
        Vector n = (v1 - v0).normalized().cross((v2 - v0).normalized() ).normalized();
        m_face_normals[i] = n;
    }
}

void EmbreeMesh::apply()
{
    // TRANSFORM VERTICES
    for(unsigned int i=0; i<m_num_vertices; i++)
    {
        m_vertices_transformed[i] = m_T * (m_vertices[i].mult_ewise(m_S));
        // also possible:
        // m_vertices_transformed[i] = matrix() * m_vertices[i];
        // might be slower
    }

    // TRANSFORM FACE NORMALS
    if(m_face_normals_transformed.size() != m_face_normals.size())
    {
        m_face_normals_transformed.resize(m_face_normals.size());
    }

    for(unsigned int i=0; i<m_face_normals.size(); i++)
    {
        auto face_normal_scaled = m_face_normals[i].mult_ewise(m_S);
        m_face_normals_transformed[i] = m_T.R * face_normal_scaled.normalized();
    }

    // TRANSFORM VERTEX NORMALS
    if(m_vertex_normals_transformed.size() != m_vertex_normals.size())
    {
        m_vertex_normals_transformed.resize(m_vertex_normals.size());
    }
    for(unsigned int i=0; i<m_vertex_normals.size(); i++)
    {
        auto vertex_normal_scaled = m_vertex_normals[i].mult_ewise(m_S);
        m_vertex_normals_transformed[i] = m_T.R * vertex_normal_scaled.normalized();
    }

    if(anyParentCommittedOnce())
    {
        rtcUpdateGeometryBuffer(m_handle, RTC_BUFFER_TYPE_VERTEX, 0);
    }
}

} // namespace rmagine