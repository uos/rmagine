#include <iostream>
#include <rmagine/util/assimp/prints.h>
#include <assimp/Importer.hpp>
#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/math/linalg.h>
#include <rmagine/util/assimp/helper.h>

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