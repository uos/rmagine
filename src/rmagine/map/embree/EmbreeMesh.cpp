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
    this->Nvertices = Nvertices;
    this->Nfaces = Nfaces;

    vertices.resize(Nvertices);

    vertices_transformed = reinterpret_cast<Vertex*>(rtcSetNewGeometryBuffer(m_handle,
                                                RTC_BUFFER_TYPE_VERTEX,
                                                0,
                                                RTC_FORMAT_FLOAT3,
                                                sizeof(Vertex),
                                                Nvertices));

    faces = reinterpret_cast<Face*>(rtcSetNewGeometryBuffer(m_handle,
                                                    RTC_BUFFER_TYPE_INDEX,
                                                    0,
                                                    RTC_FORMAT_UINT3,
                                                    sizeof(Face),
                                                    Nfaces));
}

void EmbreeMesh::init(
    const aiMesh* amesh)
{
    Nvertices = amesh->mNumVertices;
    Nfaces = amesh->mNumFaces;

    const aiVector3D* ai_vertices = amesh->mVertices;
    int num_vertices = amesh->mNumVertices;

    const aiFace* ai_faces = amesh->mFaces;
    int num_faces = amesh->mNumFaces;

    vertices.resize(Nvertices);

    vertices_transformed = reinterpret_cast<Vertex*>(rtcSetNewGeometryBuffer(m_handle,
                                                RTC_BUFFER_TYPE_VERTEX,
                                                0,
                                                RTC_FORMAT_FLOAT3,
                                                sizeof(Vertex),
                                                Nvertices));

    faces = reinterpret_cast<Face*>(rtcSetNewGeometryBuffer(m_handle,
                                                    RTC_BUFFER_TYPE_INDEX,
                                                    0,
                                                    RTC_FORMAT_UINT3,
                                                    sizeof(Face),
                                                    Nfaces));

    // copy mesh to embree buffers
    for(unsigned int i=0; i<num_vertices; i++)
    {
        vertices[i] = convert(ai_vertices[i]);
    }

    for(int i=0; i<num_faces; i++)
    {
        faces[i] = {ai_faces[i].mIndices[0], ai_faces[i].mIndices[1], ai_faces[i].mIndices[2]};
    }

    if(amesh->HasNormals())
    {
        vertex_normals.resize(Nvertices);
        vertex_normals_transformed.resize(Nvertices);
        for(size_t i=0; i<Nvertices; i++)
        {
            vertex_normals[i] = convert(amesh->mNormals[i]);
        }
    }

    computeFaceNormals();

    apply();
}

void EmbreeMesh::computeFaceNormals()
{
    face_normals.resize(Nfaces);
    face_normals_transformed.resize(Nfaces);
    for(size_t i=0; i<Nfaces; i++)
    {
        const Vector v0 = vertices[faces[i].v0];
        const Vector v1 = vertices[faces[i].v1];
        const Vector v2 = vertices[faces[i].v2];
        Vector n = (v1 - v0).normalized().cross((v2 - v0).normalized() ).normalized();
        face_normals[i] = n;
    }
}

void EmbreeMesh::apply()
{
    for(unsigned int i=0; i<Nvertices; i++)
    {
        vertices_transformed[i] = m_T * (vertices[i].mult_ewise(m_S));
    }

    for(unsigned int i=0; i<face_normals.size(); i++)
    {
        auto face_normal_scaled = face_normals[i].mult_ewise(m_S);
        face_normals_transformed[i] = m_T.R * face_normal_scaled.normalized();
    }

    for(unsigned int i=0; i<vertex_normals.size(); i++)
    {
        auto vertex_normal_scaled = vertex_normals[i].mult_ewise(m_S);
        vertex_normals_transformed[i] = m_T.R * vertex_normal_scaled.normalized();
    }

    if(anyParentCommittedOnce())
    {
        rtcUpdateGeometryBuffer(m_handle, RTC_BUFFER_TYPE_VERTEX, 0);
    }
    // bool parent_updated_once = false;

    // for(auto parent : parents)
    // {
    //     if(parent_ptr->committedOnce())
    //     {
    //         parent_updated_once = true;
    //         break;
    //     }
    // }

    // if(EmbreeScenePtr parent_ptr = parent.lock())
    // {
    //     if(parent_ptr->committedOnce())
    //     {
    //         rtcUpdateGeometryBuffer(m_handle, RTC_BUFFER_TYPE_VERTEX, 0);
    //     }
    // }
}

} // namespace rmagine