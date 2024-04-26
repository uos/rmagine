#include "rmagine/map/embree/EmbreeMesh.hpp"

// other internal deps
#include "rmagine/map/embree/EmbreeDevice.hpp"
#include "rmagine/map/embree/EmbreeScene.hpp"
#include "rmagine/map/embree/EmbreeInstance.hpp"

#include <iostream>

#include <map>
#include <cassert>



#include <rmagine/math/assimp_conversions.h>



namespace rmagine {

Point closestPointTriangle(
    const Point& p, 
    const Point& a, 
    const Point& b, 
    const Point& c)
{
  const Vector ab = b - a;
  const Vector ac = c - a;
  const Vector ap = p - a;

  const float d1 = ab.dot(ap);
  const float d2 = ac.dot(ap);
  if (d1 <= 0.f && d2 <= 0.f) 
  {
    return a;
  }

  const Vector bp = p - b;
  const float d3 = ab.dot(bp);
  const float d4 = ac.dot(bp);
  if (d3 >= 0.f && d4 <= d3) 
  {
    return b;
  }

  const Vector cp = p - c;
  const float d5 = ab.dot(cp);
  const float d6 = ac.dot(cp);
  if (d6 >= 0.f && d5 <= d6) 
  {
    return c;
  }

  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f)
  {
      const float v = d1 / (d1 - d3);
      return a + ab * v;
  }
  
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f)
  {
      const float v = d2 / (d2 - d6);
      return a + ac * v;
  }
  
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f)
  {
      const float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      return b + (c - b) * v;
  }

  const float denom = 1.f / (va + vb + vc);
  const float v = vb * denom;
  const float w = vc * denom;
  return a + ab * v + ac * w;
}


bool closestPointFunc(RTCPointQueryFunctionArguments* args)
{
    assert(args->userPtr);
    const unsigned int geomID = args->geomID;
    const unsigned int primID = args->primID;

    RTCPointQueryContext* context = args->context;
    const unsigned int stackSize = args->context->instStackSize;
    const unsigned int stackPtr = stackSize-1;

    EmbreePointQueryUserData* userData = (EmbreePointQueryUserData*)args->userPtr;

    // query position in world space
    const Vector q{args->query->x, args->query->y, args->query->z};

    /*
    * Get triangle information in global space
    */
    const EmbreeScene* scene = userData->scene;
    const EmbreeMeshPtr mesh = scene->getAs<EmbreeMesh>(geomID);
    
    // Alex: I assume it can never happen that it is no mesh since the function is only used for point queries in meshes
    const Face face = mesh->faces()[primID];
    const Vector face_normal = mesh->faceNormalsTransformed()[primID];

    const Vertex v0 = mesh->verticesTransformed()[face.v0];
    const Vertex v1 = mesh->verticesTransformed()[face.v1];
    const Vertex v2 = mesh->verticesTransformed()[face.v2];

    const Vector p = closestPointTriangle(q, v0, v1, v2);

    const float d = (p - q).l2norm();

    if (d < args->query->radius)
    {
        args->query->radius = d;
        if(d < userData->result->d)
        {
            userData->result->d = d;
            userData->result->geomID = geomID;
            userData->result->primID = primID;
            userData->result->p = p;
            userData->result->n = face_normal;
        }
        return true; // Return true to indicate that the query radius changed.
    }

    return false;
}

EmbreeMesh::EmbreeMesh(EmbreeDevicePtr device)
:Base(device)
{
    m_handle = rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_TRIANGLE);

    rtcSetGeometryPointQueryFunction(m_handle, closestPointFunc);
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

MemoryView<const Vertex, RAM> EmbreeMesh::faceNormalsTransformed() const
{
    return MemoryView<const Vertex, RAM>(m_face_normals_transformed.raw(), m_num_faces);
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
        Vector n = (v1 - v0).normalize().cross((v2 - v0).normalize() ).normalize();
        m_face_normals[i] = n;
    }
}

void EmbreeMesh::apply()
{
    // TRANSFORM VERTICES
    for(unsigned int i=0; i<m_num_vertices; i++)
    {
        m_vertices_transformed[i] = m_T * (m_vertices[i].multEwise(m_S));
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
        auto face_normal_scaled = m_face_normals[i].multEwise(m_S);
        m_face_normals_transformed[i] = m_T.R * face_normal_scaled.normalize();
    }

    // TRANSFORM VERTEX NORMALS
    if(m_vertex_normals_transformed.size() != m_vertex_normals.size())
    {
        m_vertex_normals_transformed.resize(m_vertex_normals.size());
    }
    for(unsigned int i=0; i<m_vertex_normals.size(); i++)
    {
        auto vertex_normal_scaled = m_vertex_normals[i].multEwise(m_S);
        m_vertex_normals_transformed[i] = m_T.R * vertex_normal_scaled.normalize();
    }

    if(anyParentCommittedOnce())
    {
        rtcUpdateGeometryBuffer(m_handle, RTC_BUFFER_TYPE_VERTEX, 0);
    }
}

// pt2ConstMember = &EmbreeMesh::closestPointFunc2; 

} // namespace rmagine