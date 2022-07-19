#include "rmagine/map/EmbreeMesh.hpp"

#include <iostream>

#include <map>
#include <cassert>

#include <rmagine/map/EmbreeDevice.hpp>
#include <rmagine/map/EmbreeScene.hpp>

namespace rmagine {

EmbreeMesh::EmbreeMesh(EmbreeDevicePtr device)
:m_device(device)
,m_handle(rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_TRIANGLE))
{
    
}

EmbreeMesh::EmbreeMesh(
    EmbreeDevicePtr device, 
    unsigned int Nvertices, 
    unsigned int Nfaces)
:m_device(device)
,m_handle(rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_TRIANGLE))
,Nvertices(Nvertices)
,Nfaces(Nfaces)
{   
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

EmbreeMesh::EmbreeMesh( 
    EmbreeDevicePtr device,
    const aiMesh* amesh)
:m_device(device)
,m_handle(rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_TRIANGLE))
,Nvertices(amesh->mNumVertices)
,Nfaces(amesh->mNumFaces)
{
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
        // mesh.vertices[i] = {ai_vertices[i].x, ai_vertices[i].y, ai_vertices[i].z};
        vertices[i].x = ai_vertices[i].x;
        vertices[i].y = ai_vertices[i].y;
        vertices[i].z = ai_vertices[i].z;
    }

    for(int i=0; i<num_faces; i++)
    {
        // mesh.faces[i] = {ai_faces[i].mIndices[0], ai_faces[i].mIndices[1], ai_faces[i].mIndices[2]};
        faces[i].v0 = ai_faces[i].mIndices[0];
        faces[i].v1 = ai_faces[i].mIndices[1];
        faces[i].v2 = ai_faces[i].mIndices[2];
    }

    normals.resize(amesh->mNumFaces);
    normals_transformed.resize(amesh->mNumFaces);
    for(size_t i=0; i<amesh->mNumFaces; i++)
    {
        const Vector v0 = vertices[faces[i].v0];
        const Vector v1 = vertices[faces[i].v1];
        const Vector v2 = vertices[faces[i].v2];
        Vector n = (v1 - v0).normalized().cross((v2 - v0).normalized() ).normalized();
        normals[i] = n;
    }

    Transform T;
    T.setIdentity();
    setTransform(T);
}

void EmbreeMesh::setTransform(const Transform& T)
{
    this->T = T;
    for(unsigned int i=0; i<Nvertices; i++)
    {
        vertices_transformed[i] = T * vertices[i];
    }

    for(unsigned int i=0; i<normals.size(); i++)
    {
        normals_transformed[i] = T.R * normals[i];
    }

    // rtcUpdateGeometryBuffer(m_handle, RTC_BUFFER_TYPE_VERTEX, 0);
}

void EmbreeMesh::setTransform(const Matrix4x4& T)
{
    Transform T2;
    T2.set(T);
    setTransform(T2);
}

Transform EmbreeMesh::transform() const
{
    return T;
}

RTCGeometry EmbreeMesh::handle() const
{
    return m_handle;
}

void EmbreeMesh::commit()
{
    rtcCommitGeometry(m_handle);
}

void EmbreeMesh::release()
{
    rtcReleaseGeometry(m_handle);
}

} // namespace rmagine