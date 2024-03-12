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
    const Vector ab = b-a;
    const Vector ac = c-a;
    const Vector ap = p-a;
    const Vector n = ab.cross(ac);

    // TODO: comment this in and test
    Matrix3x3 R;
    R(0,0) = ab.x;
    R(1,0) = ab.y;
    R(2,0) = ab.z;
    
    R(0,1) = ac.x;
    R(1,1) = ac.y;
    R(2,1) = ac.z;

    R(0,2) = n.x;
    R(1,2) = n.y;
    R(2,2) = n.z;

    Matrix3x3 Rinv = R.inv();
    Vector p_in_t = Rinv * ap;
    float p0 = p_in_t.x;
    float p1 = p_in_t.y;

    // Instead of this
    const bool on_ab_edge = (p0 >= 0.f && p0 <= 1.f);
    const bool on_ac_edge = (p1 >= 0.f && p1 <= 1.f);

    if(on_ab_edge && on_ac_edge)
    {
        // in triangle
        return ab * p0 + ac * p1 + a;
    } 
    else if(on_ab_edge && !on_ac_edge)
    {
        // nearest point on edge (ab)
        return ab * p0 + a;
    }
    else if(!on_ab_edge && on_ac_edge)
    {
        // nearest point on edge (ac)
        return ac * p1 + a;
    }
    else
    {
        // nearest vertex
        float d_ap = ap.l2normSquared();
        float d_bp = (p - b).l2normSquared();
        float d_cp = (p - c).l2normSquared();
        
        if(d_ap < d_bp && d_ap < d_cp)
        {
            // a best
            return a;
        } 
        else if(d_bp < d_cp) 
        { 
            // b best
            return b;
        } 
        else 
        {
            // c best
            return c;
        }
    }
}

bool closestPointFunc(RTCPointQueryFunctionArguments* args)
{
    assert(args->userPtr);
    const unsigned int geomID = args->geomID;
    const unsigned int primID = args->primID;

    RTCPointQueryContext* context = args->context;
    const unsigned int stackSize = args->context->instStackSize;
    const unsigned int stackPtr = stackSize-1;

    PointQueryUserData* userData = (PointQueryUserData*)args->userPtr;

    // query position in world space
    Vector q{args->query->x, args->query->y, args->query->z};

    /*
    * Get triangle information in global space
    */
    EmbreeScenePtr scene = *userData->scene;
    EmbreeMeshPtr mesh = scene->getAs<EmbreeMesh>(geomID);
    
    if(mesh)
    {
        Face face = mesh->faces()[primID];
        Vector face_normal = mesh->faceNormalsTransformed()[primID];

        Vertex v0 = mesh->verticesTransformed()[face.v0];
        Vertex v1 = mesh->verticesTransformed()[face.v1];
        Vertex v2 = mesh->verticesTransformed()[face.v2];

        const Vector p = closestPointTriangle(q, v0, v1, v2);

        float d = (p - q).l2norm();

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
    } else {
        EmbreeInstancePtr inst = scene->getAs<EmbreeInstance>(geomID);

        if(inst)
        {
            std::cout << "INSTANCING NOT SUPPORTED FOR POINTQUERIES YET!" << std::endl;
            // inst->scene()
        } else {
            std::cout << "WARNING: " << geomID << " unknown type" << std::endl;
        }
        
    }

    return false;
}

bool EmbreeMesh::closestPointFunc2(RTCPointQueryFunctionArguments* args)
{
    std::cout << "closestPointFunc2 called!" << std::endl;
    return true;
}

EmbreeMesh::EmbreeMesh(EmbreeDevicePtr device)
:Base(device)
{
    m_handle = rtcNewGeometry(device->handle(), RTC_GEOMETRY_TYPE_TRIANGLE);

    rtcSetGeometryPointQueryFunction(m_handle, closestPointFunc);

    // rtcSetGeometryPointQueryFunction(m_handle, [](RTCPointQueryFunctionArguments* args)
    // {
    //     assert(args->userPtr);
    //     const unsigned int geomID = args->geomID;
    //     const unsigned int primID = args->primID;
    //     RTCPointQueryContext* context = args->context;
    //     const unsigned int stackSize = args->context->instStackSize;
    //     const unsigned int stackPtr = stackSize-1;

    //     PointQueryUserData* userData = (PointQueryUserData*)args->userPtr;

    //     // query position in world space
    //     Vector q{args->query->x, args->query->y, args->query->z};

    //     // std::cout << "closestPointFunc called in " << name << std::endl;

    //     std::cout << geomID << " " << primID << std::endl;
        
    //     return true; 
    // });

    // rtcSetGeometryPointQueryFunction(m_handle, closestPointFunc2);
    // bool (RTCPointQueryFunctionArguments*) bla = boost::bind( &EmbreeMesh::closestPointFunc2, this, _1 );
    // boost::function<bool (RTCPointQueryFunctionArguments*)> func( 
    //     boost::bind( &EmbreeMesh::closestPointFunc2, this, _1 ) );

    // m_closest_point_func = boost::bind( &EmbreeMesh::closestPointFunc2, this, _1 );

    // m_closest_point_func_raw = m_closest_point_func.target<RTCPointQueryFunction>();
    
    // rtcSetGeometryPointQueryFunction(m_handle, &m_closest_point_func);

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