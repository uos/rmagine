#include "imagine/map/EmbreeMap.hpp"

#include <iostream>
#include <Eigen/Dense>

namespace imagine {

Eigen::Vector3f closestPointTriangle(
    const Eigen::Vector3f& p, 
    const Eigen::Vector3f& a, 
    const Eigen::Vector3f& b, 
    const Eigen::Vector3f& c)
{
    const Eigen::Vector3f ab = b-a;
    const Eigen::Vector3f ac = c-a;
    const Eigen::Vector3f ap = p-a;
    const Eigen::Vector3f n = ab.cross(ac);

    Eigen::Matrix3f R;
    R.col(0) = ab;
    R.col(1) = ac;
    R.col(2) = n;
    const Eigen::Matrix3f Rinv = R.inverse();
    const Eigen::Vector3f p_in_t = Rinv * ap;

    // p0 * ab + p1 * ac + p2 * n == ap
    // projection: p0 * ab + p1 * ac
    float p0 = p_in_t.x();
    float p1 = p_in_t.y();

    const bool on_ab_edge = (p0 >= 0.f && p0 <= 1.f);
    const bool on_ac_edge = (p1 >= 0.f && p1 <= 1.f);

    if(on_ab_edge && on_ac_edge)
    {
        // in triangle
        return p0 * ab + p1 * ac + a;
    } 
    else if(on_ab_edge && !on_ac_edge)
    {
        // nearest point on edge (ab)
        return p0 * ab + a;
    }
    else if(!on_ab_edge && on_ac_edge)
    {
        // nearest point on edge (ac)
        return p1 * ac + a;
    }
    else
    {
        // nearest vertex
        float d_ap = ap.squaredNorm();
        float d_bp = (p - b).squaredNorm();
        float d_cp = (p - c).squaredNorm();
        
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

Point closestPointTriangle(
    const Point& p, 
    const Point& a, 
    const Point& b, 
    const Point& c)
{
    const Vector ab = b-a;
    const Vector ac = c-a;
    const Vector ap = p-a;
    const Vector n = cross(ab, ac);

    Eigen::Matrix3f R;
    R.col(0) = Eigen::Vector3f(ab.x, ab.y, ab.z);
    R.col(1) = Eigen::Vector3f(ac.x, ac.y, ac.z);
    R.col(2) = Eigen::Vector3f(n.x, n.y, n.z);
    
    const Eigen::Matrix3f Rinv = R.inverse();
    const Eigen::Vector3f p_in_t = Rinv * Eigen::Vector3f(ap.x, ap.y, ap.z);

    // p0 * ab + p1 * ac + p2 * n == ap
    // projection: p0 * ab + p1 * ac
    float p0 = p_in_t.x();
    float p1 = p_in_t.y();

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
        float d_ap = l2normSquared(ap);
        float d_bp = l2normSquared(p - b);
        float d_cp = l2normSquared(p - c);
        
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
    unsigned int* face_view = &mesh_part.faces[primID *  3];

    float* v1_view = &mesh_part.vertices[face_view[0] * 3];
    float* v2_view = &mesh_part.vertices[face_view[1] * 3];
    float* v3_view = &mesh_part.vertices[face_view[2] * 3];

    Point v1{v1_view[0], v1_view[1], v1_view[2]};
    Point v2{v2_view[0], v2_view[1], v2_view[2]};
    Point v3{v3_view[0], v3_view[1], v3_view[2]};

    const Vector p = closestPointTriangle(q, v1, v2, v3);

    float d = l2norm(p - q);

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

EmbreeMap::EmbreeMap(const aiScene* ascene)
{
    initializeDevice();
    scene = rtcNewScene(device);
    for(unsigned int mesh_id = 0; mesh_id < ascene->mNumMeshes; mesh_id++)
    {
        const aiMesh* mesh = ascene->mMeshes[mesh_id];
        RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
        
        const aiVector3D* ai_vertices = mesh->mVertices;
        int num_vertices = mesh->mNumVertices;

        const aiFace* ai_faces = mesh->mFaces;
        int num_faces = mesh->mNumFaces;

        EmbreeMesh part;

        part.vertices = (float*) rtcSetNewGeometryBuffer(geom,
                                                        RTC_BUFFER_TYPE_VERTEX,
                                                        0,
                                                        RTC_FORMAT_FLOAT3,
                                                        3*sizeof(float),
                                                        num_vertices);

        part.faces = (unsigned*) rtcSetNewGeometryBuffer(geom,
                                                                RTC_BUFFER_TYPE_INDEX,
                                                                0,
                                                                RTC_FORMAT_UINT3,
                                                                3*sizeof(unsigned),
                                                                num_faces);

        if (part.vertices && part.faces)
        {
            // copy mesh to embree buffers
            for(int i=0; i<num_vertices; i++)
            {
                part.vertices[i*3+0] = ai_vertices[i].x;
                part.vertices[i*3+1] = ai_vertices[i].y;
                part.vertices[i*3+2] = ai_vertices[i].z;
            }

            for(int i=0; i<num_faces; i++)
            {
                part.faces[i*3+0] = ai_faces[i].mIndices[0];
                part.faces[i*3+1] = ai_faces[i].mIndices[1];
                part.faces[i*3+2] = ai_faces[i].mIndices[2];
            }

            part.normals.resize(mesh->mNumFaces * 3);
            for(size_t i=0; i<mesh->mNumFaces; i++)
            {
                unsigned int v0_id = ai_faces[i].mIndices[0];
                unsigned int v1_id = ai_faces[i].mIndices[1];
                unsigned int v2_id = ai_faces[i].mIndices[2];

                const Vector v0{ai_vertices[v0_id].x, ai_vertices[v0_id].y, ai_vertices[v0_id].z};
                const Vector v1{ai_vertices[v1_id].x, ai_vertices[v1_id].y, ai_vertices[v1_id].z};
                const Vector v2{ai_vertices[v2_id].x, ai_vertices[v2_id].y, ai_vertices[v2_id].z};
                
                Vector n = normalized( cross(normalized(v1 - v0), normalized(v2 - v0) ) );

                part.normals[i * 3 + 0] = n.x;
                part.normals[i * 3 + 1] = n.y;
                part.normals[i * 3 + 2] = n.z;
            }

            parts.push_back(part);
        } else {
            std::cerr << "could not create embree vertices or faces for mesh id " << mesh_id << std::endl;
            // continue;
        }

        // RTCPointQueryFunction cb = std::bind(&EmbreeMesh::closestPointFunc, this);
        rtcSetGeometryPointQueryFunction(geom, closestPointFunc);
        rtcCommitGeometry(geom);
        rtcAttachGeometry(scene, geom);
        rtcReleaseGeometry(geom);
    }

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
    user_data.parts = &parts;
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