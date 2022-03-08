#include "rmagine/map/EmbreeMap.hpp"

#include <iostream>

#include <map>
#include <cassert>

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

    // Eigen::Matrix3f R;
    // R.col(0) = Eigen::Vector3f(ab.x, ab.y, ab.z);
    // R.col(1) = Eigen::Vector3f(ac.x, ac.y, ac.z);
    // R.col(2) = Eigen::Vector3f(n.x, n.y, n.z);
    
    // const Eigen::Matrix3f Rinv = R.inverse();
    // const Eigen::Vector3f p_in_t = Rinv * Eigen::Vector3f(ap.x, ap.y, ap.z);

    // // p0 * ab + p1 * ac + p2 * n == ap
    // // projection: p0 * ab + p1 * ac
    // float p0 = p_in_t.x();
    // float p1 = p_in_t.y();

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
    // std::cout << "closestPointFunc called." << std::endl;
    assert(args->userPtr);
    const unsigned int geomID = args->geomID;
    const unsigned int primID = args->primID;

    RTCPointQueryContext* context = args->context;
    const unsigned int stackSize = args->context->instStackSize;
    const unsigned int stackPtr = stackSize-1;

    PointQueryUserData* userData = (PointQueryUserData*)args->userPtr;

    // query position in world space
    Vector q{args->query->x, args->query->y, args->query->z};

    // std::cout << "- Query: " << q.transpose() << std::endl;
    
    /*
    * Get triangle information in global space
    */
    const EmbreeMesh& mesh_part = userData->parts->at(geomID);

    // Triangle const& t = triangle_mesh->triangles[primID];
    Face face = mesh_part.faces[primID];

    Vertex v0 = mesh_part.vertices[face.v0];
    Vertex v1 = mesh_part.vertices[face.v1];
    Vertex v2 = mesh_part.vertices[face.v2];

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
        }
        return true; // Return true to indicate that the query radius changed.
    }

    return false;
}

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
    printf("error %d: %s\n", error, str);
}

void print(const aiMatrix4x4& T)
{
    std::cout << T.a1 << " " << T.a2 << " " << T.a3 << " " << T.a4 << std::endl;
    std::cout << T.b1 << " " << T.b2 << " " << T.b3 << " " << T.b4 << std::endl;
    std::cout << T.c1 << " " << T.c2 << " " << T.c3 << " " << T.c4 << std::endl;
    std::cout << T.d1 << " " << T.d2 << " " << T.d3 << " " << T.d4 << std::endl;
}

void convert(const aiMatrix4x4& aT, Matrix4x4& T)
{
    // TODO: check
    T(0,0) = aT.a1;
    T(0,1) = aT.a2;
    T(0,2) = aT.a3;
    T(0,3) = aT.a4;
    T(1,0) = aT.b1;
    T(1,1) = aT.b2;
    T(1,2) = aT.b3;
    T(1,3) = aT.b4;
    T(2,0) = aT.c1;
    T(2,1) = aT.c2;
    T(2,2) = aT.c3;
    T(2,3) = aT.c4;
    T(3,0) = aT.d1;
    T(3,1) = aT.d2;
    T(3,2) = aT.d3;
    T(3,3) = aT.d4;
}

EmbreeMap::EmbreeMap(const aiScene* ascene)
{
    initializeDevice();

    scene = rtcNewScene(device);

    std::map<unsigned int, Matrix4x4> Tmap;

    // Parsing transformation tree
    unsigned int geom_id = 0;
    const aiNode* root_node = ascene->mRootNode;
    for(unsigned int i=0; i<root_node->mNumChildren; i++)
    {
        const aiNode* n = root_node->mChildren[i];
        if(n->mNumChildren == 0)
        {
            // Leaf
            if(n->mNumMeshes > 0)
            {
                aiMatrix4x4 aT = n->mTransformation;
                Matrix4x4 T;
                convert(aT, T);
                Tmap[n->mMeshes[0]] = T;
            }
        } else {
            // TODO: handle deeper tree. concatenate transformations
            // std::cout << "- Children: " << n->mNumChildren << std::endl;
        }
    }

    for(unsigned int mesh_id = 0; mesh_id < ascene->mNumMeshes; mesh_id++)
    {
        // std::cout << "Mesh " << mesh_id << std::endl;
        const aiMesh* amesh = ascene->mMeshes[mesh_id];

        const aiVector3D* ai_vertices = amesh->mVertices;
        int num_vertices = amesh->mNumVertices;

        const aiFace* ai_faces = amesh->mFaces;
        int num_faces = amesh->mNumFaces;

        if(num_faces == 0)
        {
            continue;
        }

        // Mesh
        EmbreeMesh mesh;
        mesh.handle = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

        mesh.Nvertices = num_vertices;

        mesh.vertices = (Vertex*) rtcSetNewGeometryBuffer(mesh.handle,
                                                        RTC_BUFFER_TYPE_VERTEX,
                                                        0,
                                                        RTC_FORMAT_FLOAT3,
                                                        sizeof(Point),
                                                        num_vertices);

        mesh.Nfaces = num_faces;

        mesh.faces = (Face*) rtcSetNewGeometryBuffer(mesh.handle,
                                                                RTC_BUFFER_TYPE_INDEX,
                                                                0,
                                                                RTC_FORMAT_UINT3,
                                                                sizeof(Face),
                                                                num_faces);

        // Memory<Vector, RAM> vertices_transformed(num_vertices);

        Matrix4x4 T;
        T.setIdentity();

        auto fit = Tmap.find(mesh_id);
        if(fit != Tmap.end())
        {
            // Found transform!
            T = fit->second;

            // std::cout << "Got Transform: " << std::endl;
            // for(size_t i=0; i<4; i++)
            // {
            //     for(size_t j=0; j<4; j++)
            //     {
            //         std::cout << T(i,j) << " ";
            //     }
            //     std::cout << std::endl;
            // }
        }

        // set_identity(mesh.T);

        // copy mesh to embree buffers
        for(unsigned int i=0; i<num_vertices; i++)
        {
            Vector v;
            v.x = ai_vertices[i].x;
            v.y = ai_vertices[i].y;
            v.z = ai_vertices[i].z;
            
            v = T * v;

            mesh.vertices[i] = v;
            
        }

        // for(unsigned int i=0; i<num_vertices; i++)
        // {
        //     mesh.vertices[i*3+0] = vertices_transformed[i].x;
        //     mesh.vertices[i*3+1] = vertices_transformed[i].y;
        //     mesh.vertices[i*3+2] = vertices_transformed[i].z;
        // }

        // mesh.bb = bb;
        for(int i=0; i<num_faces; i++)
        {
            mesh.faces[i].v0 = ai_faces[i].mIndices[0];
            mesh.faces[i].v1 = ai_faces[i].mIndices[1];
            mesh.faces[i].v2 = ai_faces[i].mIndices[2];
        }

        mesh.normals.resize(amesh->mNumFaces);
        for(size_t i=0; i<amesh->mNumFaces; i++)
        {
            unsigned int v0_id = ai_faces[i].mIndices[0];
            unsigned int v1_id = ai_faces[i].mIndices[1];
            unsigned int v2_id = ai_faces[i].mIndices[2];

            const Vector v0 = mesh.vertices[v0_id];
            const Vector v1 = mesh.vertices[v1_id];
            const Vector v2 = mesh.vertices[v2_id];

            Vector n = (v1 - v0).normalized().cross((v2 - v0).normalized() ).normalized();

            mesh.normals[i] = n;
        }

        meshes.push_back(mesh);
    }

    if(meshes.size() == 0)
    {
        throw std::runtime_error("No meshes!");
    }

    // Add everything to embree

    // 1. Meshes
    for(unsigned int i = 0; i<meshes.size(); i++)
    {
        EmbreeMesh& mesh = meshes[i];
        rtcSetGeometryPointQueryFunction(mesh.handle, closestPointFunc);
        rtcAttachGeometryByID(scene, mesh.handle, i);
        rtcReleaseGeometry(mesh.handle);
        // rtcSetGeometryTransform(mesh.handle, 1.0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, reinterpret_cast<float*>(&mesh.T) );
        rtcCommitGeometry(mesh.handle);
    }

    // TODO
    // // 2. Instances
    // for(unsigned int i=0; i<instances.size(); i++)
    // {
    //     EmbreeInstance& instance = instances[i];

    //     rtcSetGeometryInstancedScene(instance.handle, scene);
    //     // new scene here?
    //     rtcAttachGeometryByID(scene, instance.handle, i + meshes.size());
    //     rtcReleaseGeometry(instance.handle);
        
    //     rtcSetGeometryTransform(instance.handle, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, reinterpret_cast<float*>(&instance.T) );
    //     rtcCommitGeometry(instance.handle);
    // }

    rtcCommitScene(scene);

    rtcInitPointQueryContext(&pq_context);
}

EmbreeMap::~EmbreeMap()
{
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

Point EmbreeMap::closestPoint(const Point& qp)
{
    RTCPointQuery query;
    query.x = qp.x; 
    query.y = qp.y;
    query.z = qp.z;
    query.radius = std::numeric_limits<float>::max();
    query.time = 0.0;

    ClosestPointResult result;

    PointQueryUserData user_data;
    // user_data.parts = &parts;
    user_data.result = &result;

    rtcPointQuery(scene, &query, &pq_context, nullptr, (void*)&user_data);

    if(result.geomID == RTC_INVALID_GEOMETRY_ID || result.primID == RTC_INVALID_GEOMETRY_ID)
    {
        throw std::runtime_error("Cannot find nearest point on surface");
    }

    return result.p;
}

void EmbreeMap::initializeDevice()
{
    device = rtcNewDevice(NULL);

    if (!device)
    {
        std::cerr << "error " << rtcGetDeviceError(NULL) << ": cannot create device" << std::endl;
    }

    rtcSetDeviceErrorFunction(device, errorFunction, NULL);
}

} // namespace mamcl