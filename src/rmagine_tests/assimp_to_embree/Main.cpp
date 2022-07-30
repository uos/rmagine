#include <iostream>
#include <rmagine/util/assimp/prints.h>
#include <assimp/Importer.hpp>
#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/math/linalg.h>

using namespace rmagine;



void printRaycast(EmbreeScenePtr scene, Vector3 orig, Vector3 dir)
{
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    RTCRayHit rayhit;
    rayhit.ray.org_x = orig.x;
    rayhit.ray.org_y = orig.y;
    rayhit.ray.org_z = orig.z;
    rayhit.ray.dir_x = dir.x;
    rayhit.ray.dir_y = dir.y;
    rayhit.ray.dir_z = dir.z;
    rayhit.ray.tnear = 0;
    rayhit.ray.tfar = INFINITY;
    rayhit.ray.mask = 0;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene->handle(), &context, &rayhit);

    std::cout << "Raycast:" << std::endl;

    std::cout << "- range: " << rayhit.ray.tfar << std::endl;

    if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
    {
        if(rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID)
        {
            auto geom = scene->get(rayhit.hit.instID[0]);
            std::cout << "- id: " << rayhit.hit.instID[0] << std::endl;
            std::cout << "- type: instance" << std::endl;
            std::cout << "- name: " << geom->name << std::endl;
        } else {
            auto geom = scene->get(rayhit.hit.geomID);
            std::cout << "- id: " << rayhit.hit.geomID << std::endl;
            std::cout << "- type: mesh" << std::endl;
            std::cout << "- name: " << geom->name << std::endl;
        }
        
    }
}


void get_nodes_with_meshes(const aiNode* node, std::vector<const aiNode*>& mesh_nodes)
{
    if(node->mNumMeshes > 0)
    {
        mesh_nodes.push_back(node);
    }

    if(node->mNumChildren > 0)
    {
        // is parent. check if it has meshes anyway
        for(size_t i=0; i<node->mNumChildren; i++)
        {
            get_nodes_with_meshes(node->mChildren[i], mesh_nodes);
        }
    }
}

std::vector<const aiNode*> get_nodes_with_meshes(const aiNode* node)
{
    std::vector<const aiNode*> ret;
    get_nodes_with_meshes(node, ret);
    return ret;
}

EmbreeScenePtr make_embree_scene(const aiScene* ascene)
{   
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();

    std::vector<EmbreeMeshPtr> meshes;

    // 1. meshes
    for(size_t i=0; i<ascene->mNumMeshes; i++)
    {
        const aiMesh* amesh = ascene->mMeshes[i];
        EmbreeMeshPtr mesh = std::make_shared<EmbreeMesh>(amesh);
        mesh->commit();
        meshes.push_back(mesh);
    }

    std::unordered_map<EmbreeMeshPtr, EmbreeScenePtr> mesh_scenes;

    std::vector<EmbreeInstancePtr> instances;

    // 2. instances
    const aiNode* root_node = ascene->mRootNode;
    std::vector<const aiNode*> mesh_nodes = get_nodes_with_meshes(root_node);

    std::cout << "Got " << mesh_nodes.size() << " nodes with meshes" << std::endl;

    for(size_t i=0; i<mesh_nodes.size(); i++)
    {
        const aiNode* node = mesh_nodes[i];
        std::cout << "- " << i << ": " << node->mName.C_Str();

        std::cout << ", path (reversed) ";

        Matrix4x4 M = convert(node->mTransformation);

        const aiNode* it = node->mParent;

        while(it != NULL)
        {
            std::cout << " -> " << it->mName.C_Str();
            Matrix4x4 M2 = convert(it->mTransformation);
            M = M2 * M;
            it = it->mParent;
        }

        std::cout << std::endl;
        
        // std::cout << "- Transformation Matrix: " << std::endl;
        // std::cout << M << std::endl;
        Transform T;
        Vector3 scale;
        decompose(M, T, scale);

        
        
        unsigned int mesh_id = node->mMeshes[0];

        
        EmbreeMeshPtr mesh = meshes[mesh_id];
        EulerAngles e;
        e.set(T.R);

        std::cout << "-- instanciate mesh " << mesh_id << ", " << mesh->name <<  " at " << T.t << ", " << e << " with scale " << scale << std::endl;

        EmbreeScenePtr mesh_scene;
        if(mesh_scenes.find(mesh) != mesh_scenes.end())
        {
            // take existing scene
            mesh_scene = mesh_scenes[mesh];
        } else {
            // new scene required
            mesh_scene = std::make_shared<EmbreeScene>();
            mesh_scenes[mesh] = mesh_scene;
            std::cout << "--- created new scene for mesh: mesh_scene" << std::endl;
        }

        mesh_scene->add(mesh);
        mesh_scene->commit();
        std::cout << "--- mesh added to mesh_scene" << std::endl;

        EmbreeInstancePtr mesh_instance = std::make_shared<EmbreeInstance>();
        mesh_instance->set(mesh_scene);
        mesh_instance->name = node->mName.C_Str();
        mesh_instance->setTransform(T);
        mesh_instance->setScale(scale);
        mesh_instance->apply();
        mesh_instance->commit();
        std::cout << "--- mesh_instance created" << std::endl;
        scene->add(mesh_instance);
    }

    std::cout << "add meshes that are not instanciated ..." << std::endl;
    // ADD MESHES THAT ARE NOT INSTANCIATED
    for(auto mesh : meshes)
    {
        if(mesh_scenes.find(mesh) == mesh_scenes.end())
        {
            // mesh was not instanciated
            scene->add(mesh);
        }
    }

    return scene;
}   

int main(int argc, char** argv)
{
    std::cout << "Assimp to embree conversion test" << std::endl;

     // minimum 1 argument
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " mesh_file" << std::endl;
        return 0;
    }

    std::string filename = argv[1];

    std::cout << "Inputs: " << std::endl;
    std::cout << "- filename: " << filename << std::endl;

    Assimp::Importer importer;
    const aiScene* ascene = importer.ReadFile(filename, 0);

    print(ascene);


    std::cout << "Start converting scene" << std::endl;

    auto scene = make_embree_scene(ascene);
    // scene->optimize();
    scene->commit();
    
    printRaycast(scene, {5.0, 2.0, 3.0}, {1.0, 0.0, 0.0});
    
    // rotated 45 degrees around z axis. (1,1) | (1,-1) | (-1,1) | (-1,-1)
    // should all give 0.2 range

    Vector3 dir{1.0, 1.0, 0.0};
    dir.normalize();
    printRaycast(scene, {5.0, 2.0, 3.0}, dir);
    printRaycast(scene, {5.0, 2.0, 3.0}, {0.0, 0.0, 1.0});

    return 0;
}