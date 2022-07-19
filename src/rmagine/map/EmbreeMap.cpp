#include "rmagine/map/EmbreeMap.hpp"

#include <rmagine/map/EmbreeScene.hpp>
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

void print(const aiMatrix4x4& T)
{
    std::cout << T.a1 << " " << T.a2 << " " << T.a3 << " " << T.a4 << std::endl;
    std::cout << T.b1 << " " << T.b2 << " " << T.b3 << " " << T.b4 << std::endl;
    std::cout << T.c1 << " " << T.c2 << " " << T.c3 << " " << T.c4 << std::endl;
    std::cout << T.d1 << " " << T.d2 << " " << T.d3 << " " << T.d4 << std::endl;
}

void convert(const aiMatrix4x4& aT, Matrix4x4& T)
{
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



EmbreeMap::EmbreeMap()
:device(new EmbreeDevice)
,scene(new EmbreeScene(device))
{
    
}

EmbreeMap::EmbreeMap(EmbreeDevicePtr device)
:device(device)
,scene(new EmbreeScene(device))
{

}


EmbreeMap::EmbreeMap(EmbreeDevicePtr device, const aiScene* ascene)
:EmbreeMap(device)
{
    set(ascene);
}

EmbreeMap::EmbreeMap(const aiScene* ascene)
:EmbreeMap()
{
    set(ascene);
}

void EmbreeMap::set(const aiScene* ascene)
{
    scene->setQuality(RTCBuildQuality::RTC_BUILD_QUALITY_LOW);
    scene->setFlags(RTCSceneFlags::RTC_SCENE_FLAG_DYNAMIC);

    meshes = loadMeshes(ascene);
    instances = loadInstances(ascene->mRootNode, meshes);

    // instancing implemented. can be enabled with this flag
    // - problem: slower runtime
    // if accelerated: how to handle object ids. Geometry ID or instance ID?
    bool instanced = true;

    if(instanced)
    {
        std::cout << "Using Embree with Instance Level" << std::endl;
        for(auto instance : instances)
        {
            // if(instance->mesh()->instances().size() == 1)
            // {

            //     std::cout << "Mesh has only one istance" << std::endl;
            
            //     auto mesh = instance->mesh();
            //     mesh->transform(instance->T);
            //     instance->T.setIdentity();
            //     mesh->setScene(scene);

            // } else {
                instance->setScene(scene);
                instance->commit();

            // }
        }

        // what to do with meshes without instance? apply them to upper geometry
        for(auto mesh : meshes)
        {
            if(mesh->instances().size() == 0)
            {
                // add mesh to the uppest scene
                mesh->setScene(scene);
            }

            if(mesh->instances().size() == 1)
            {
                // delete
            }
        }
    } else {
        std::cout << "Using Embree without Instance Level" << std::endl;
        // transform each mesh
        for(auto mesh : meshes)
        {
            if(mesh->instances().size() == 1)
            {
                auto instance = *(mesh->instances().begin());
                mesh->transform(instance->T);
                instance->T.setIdentity();
            } 
            else if(mesh->instances().size() > 1) 
            {
                std::cout << "Mesh has more than one instances. Instanced built is required!" << std::endl;
            }

            // scene->add(mesh);
            mesh->setScene(scene);
        }
    }

    scene->commit();
    rtcInitPointQueryContext(&pq_context);
}

unsigned int EmbreeMap::addMesh(EmbreeMeshPtr mesh)
{
    unsigned int id = meshes.size();

    mesh->setScene(scene);
    meshes.push_back(mesh);

    return id;
}

EmbreeMap::~EmbreeMap()
{
    
}

Point EmbreeMap::closestPoint(const Point& qp)
{
    std::cout << "TODO: check if closestPoint is working after refactoring" << std::endl;
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

    rtcPointQuery(scene->handle(), &query, &pq_context, nullptr, (void*)&user_data);

    if(result.geomID == RTC_INVALID_GEOMETRY_ID || result.primID == RTC_INVALID_GEOMETRY_ID)
    {
        throw std::runtime_error("Cannot find nearest point on surface");
    }

    return result.p;
}

std::vector<EmbreeMeshPtr> EmbreeMap::loadMeshes(const aiScene* ascene)
{
    std::vector<EmbreeMeshPtr> meshes;
    for(unsigned int mesh_id = 0; mesh_id < ascene->mNumMeshes; mesh_id++)
    {
        const aiMesh* amesh = ascene->mMeshes[mesh_id];
        EmbreeMeshPtr mesh(new EmbreeMesh(device, amesh));
        mesh->commit();
        meshes.push_back(mesh);
    }

    return meshes;
}

std::vector<EmbreeInstancePtr> EmbreeMap::loadInstances(
    const aiNode* root_node,
    std::vector<EmbreeMeshPtr>& meshes)
{
    std::vector<EmbreeInstancePtr> instances;

    for(unsigned int i=0; i<root_node->mNumChildren; i++)
    {
        const aiNode* n = root_node->mChildren[i];
        if(n->mNumChildren == 0)
        {
            // Leaf
            if(n->mNumMeshes > 0)
            {
                EmbreeInstancePtr instance(new EmbreeInstance(device));
                
                // convert assimp matrix to internal type
                convert(n->mTransformation, instance->T);
                unsigned int mesh_id = n->mMeshes[0];
                // instance.

                // get mesh to be instanced
                auto mesh = meshes[mesh_id];
                // make one scene per mesh 
                // EmbreeScenePtr mesh_scene(new EmbreeScene(device) );
                if(!mesh->scene())
                {
                    mesh->setNewScene();
                }
                mesh->scene()->commit();

                // connect mesh geometry to instance and instance to geometry
                instance->setMesh(mesh);
                mesh->addInstance(instance);

                // commit
                instance->commit();

                // attach to scene
                // instance->setScene(scene);
                instances.push_back(instance);
            }

        } else {
            // TODO: handle deeper tree. concatenate transformations
            std::cout << "Found instance tree in map. Currently not supported." << std::endl;
            std::cout << "- Children: " << n->mNumChildren << std::endl;
        }
    }

    return instances;
}

} // namespace rmagine