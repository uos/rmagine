#include <iostream>
#include <sstream>

// General rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/synthetic.h>

#include <rmagine/map/EmbreeMap.hpp>

#include <rmagine/map/embree/embree_shapes.h>
#include <rmagine/util/prints.h>

using namespace rmagine;

void scene_1()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();
    
    {
        // add sphere. sphere should stay alive as long as the scene lives
        EmbreeMeshPtr sphere = std::make_shared<EmbreeSphere>(1.0);
        std::cout << sphere->scale() << std::endl;
        scene->add(sphere);
    }
    std::cout << "now sphere should be alive" << std::endl;

    scene.reset();
    std::cout << "Scene destroyed." << std::endl;
}

void scene_2()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();

    unsigned int sphere_id = 0;
    {
        // add sphere. sphere should stay alive as long as the scene lives
        EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>(1.0);
        std::cout << sphere->scale() << std::endl;
        sphere_id = scene->add(sphere);
    }

    std::cout << "now sphere should be alive" << std::endl;

    // save mesh
    EmbreeGeometryPtr sphere = scene->geometries()[sphere_id];

    scene.reset();
    std::cout << "Now Scene should be destroyed but not the sphere" << std::endl;

    if(!sphere->parent.lock())
    {
        std::cout << "- sphere noticed parent was destroyed." << std::endl;
    }
}

void scene_3()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();
    unsigned int sphere_id = 0;
    {
        // add sphere. sphere should stay alive as long as the scene lives
        EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>(1.0);
        std::cout << sphere->scale() << std::endl;
        sphere_id = scene->add(sphere);
    }

    EmbreeGeometryPtr sphere = scene->remove(sphere_id);
    

    std::cout << "Removed mesh from scene. the next numbers should be 0" << std::endl;
    std::cout << scene->count<EmbreeMesh>() << std::endl;
    std::cout << sphere->parent.lock() << std::endl;
}

void scene_4()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();
    
    
    // add sphere. sphere should stay alive as long as the scene lives
    EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>(1.0);
    std::cout << sphere->scale() << std::endl;
    scene->add(sphere);
    

    // the scenes mesh should get invalid when destroying mesh
    // sphere.reset();
    std::cout << "meshes: " << scene->count<EmbreeMesh>() << std::endl;

    sphere->commit();
    scene->commit();

    RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    RTCRayHit rayhit;
    rayhit.ray.org_x = 0;
    rayhit.ray.org_y = 0;
    rayhit.ray.org_z = 0;
    rayhit.ray.dir_x = 1;
    rayhit.ray.dir_y = 0;
    rayhit.ray.dir_z = 0;
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
        
        std::cout << "- geomID: " << rayhit.hit.geomID << std::endl;
    }

    if(rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID)
    {
        std::cout << "- instID: " << rayhit.hit.instID[0] << std::endl;
    }
    
}

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

    if(rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID)
    {
        std::cout << "- instID: " << rayhit.hit.instID[0] << std::endl;
    }
}

void scene_5()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();

    {
        std::cout << "Generate Sphere" << std::endl;
        EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>(1.0);
        Transform T;
        T.setIdentity();
        T.t.y = 5.0;

        sphere->setTransform(T);
        sphere->apply();
        sphere->commit();

        scene->add(sphere);
    }

    {
        std::cout << "Generate Cube" << std::endl;
        EmbreeCubePtr cube = std::make_shared<EmbreeCube>();
        Transform T;
        T.setIdentity();
        T.t.y = -5.0;
        cube->setTransform(T);

        cube->apply();
        cube->commit();

        scene->add(cube);
    }

    std::cout << "Commit Scene" << std::endl;
    scene->commit();

    std::cout << "Raycast.." << std::endl;
    printRaycast(scene, {0.0, 5.0, 0.0}, {1.0, 0.0, 0.0});
    printRaycast(scene, {0.0, -5.0, 0.0}, {1.0, 0.0, 0.0});
}

void scene_6()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();


    { // CUBE INSTANCE
        EmbreeCubePtr cube = std::make_shared<EmbreeCube>();
        cube->commit();

        // make cube scene
        EmbreeScenePtr cube_scene = std::make_shared<EmbreeScene>();
        cube_scene->add(cube);
        cube_scene->commit();

        // make cube scene instance
        EmbreeInstancePtr cube_inst = std::make_shared<EmbreeInstance>();
        cube_inst->set(cube_scene);

        Transform T;
        T.setIdentity();
        T.t.y = -5.0;
        Matrix4x4 M;
        M.set(T);

        cube_inst->setTransform(M);
        cube_inst->apply();

        cube_inst->commit();

        scene->add(cube_inst);
    }

    { // SPHERE MESH
        std::cout << "Generate Sphere" << std::endl;
        EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>(1.0);
        Transform T;
        T.setIdentity();
        T.t.y = 5.0;

        sphere->setTransform(T);
        sphere->apply();
        sphere->commit();

        scene->add(sphere);
    }

    std::cout << "Commit Scene" << std::endl;
    scene->commit();

    std::cout << "Raycast.." << std::endl;
    printRaycast(scene, {0.0, 5.0, 0.0}, {1.0, 0.0, 0.0});
    printRaycast(scene, {0.0, -5.0, 0.0}, {1.0, 0.0, 0.0});
}

void scene_7()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();

    std::cout << "Generate Sphere" << std::endl;
    // gen sphere mesh
    EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>(1.0);
    sphere->commit();

    // make sphere scene
    EmbreeScenePtr sphere_scene = std::make_shared<EmbreeScene>();
    sphere_scene->add(sphere);
    sphere_scene->commit();

    // make N sphere instances
    int Ninstances = 100;
    for(int i=0; i < Ninstances; i++)
    {
        EmbreeInstancePtr sphere_inst = std::make_shared<EmbreeInstance>();
        sphere_inst->set(sphere_scene);

        float t = static_cast<float>(i - Ninstances / 2);

        Transform T;
        T.setIdentity();
        T.t.y = t;

        Vector3 scale = {
            0.01f * static_cast<float>(i + 1),
            0.01f * static_cast<float>(i + 1),
            0.01f * static_cast<float>(i + 1)
        };

        sphere_inst->setTransform(T);
        sphere_inst->setScale(scale);
        sphere_inst->apply();
        sphere_inst->commit();

        scene->add(sphere_inst);
    }

    std::cout << "Commit Scene" << std::endl;
    scene->commit();

    std::cout << "Raycast.." << std::endl;
    printRaycast(scene, {0.0, 5.0, 0.0}, {1.0, 0.0, 0.0});
    printRaycast(scene, {0.0, -5.0, 0.0}, {1.0, 0.0, 0.0});
} 

void scene_8()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();

    std::cout << "Generate Sphere" << std::endl;
    // gen sphere mesh
    EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>(1.0);
    sphere->name = "Sphere Mesh";
    sphere->commit();

    // make sphere scene
    EmbreeScenePtr sphere_scene = std::make_shared<EmbreeScene>();
    sphere_scene->add(sphere);
    sphere_scene->commit();

    // make N sphere instances
    int Ninstances = 100;
    for(int i=0; i < Ninstances; i++)
    {
        EmbreeInstancePtr sphere_inst = std::make_shared<EmbreeInstance>();
        sphere_inst->set(sphere_scene);

        float t = static_cast<float>(i - Ninstances / 2);

        Transform T;
        T.setIdentity();
        T.t.y = t;

        Vector3 scale = {
            0.01f * static_cast<float>(i + 1),
            0.01f * static_cast<float>(i + 1),
            0.01f * static_cast<float>(i + 1)
        };
        
        std::stringstream ss;
        ss << "Sphere Instance " << i;
        sphere_inst->name = ss.str();
        sphere_inst->setTransform(T);
        sphere_inst->setScale(scale);
        sphere_inst->apply();
        sphere_inst->commit();

        scene->add(sphere_inst);
    }

    unsigned int cube_id = 0;
    { // CUBE MESH
        EmbreeCubePtr cube = std::make_shared<EmbreeCube>();
        cube->name = "Cube";

        Transform T;
        T.setIdentity();
        T.t.x += 10.0;
        cube->setTransform(T);
        cube->apply();
        cube->commit();

        cube_id = scene->add(cube);
    }

    std::cout << "Commit Scene" << std::endl;
    scene->commit();

    std::cout << "Raycast.." << std::endl;
    printRaycast(scene, {0.0, 5.0, 0.0}, {1.0, 0.0, 0.0});
    printRaycast(scene, {0.0, -5.0, 0.0}, {1.0, 0.0, 0.0});

    // try to hit cube
    printRaycast(scene, {5.0, 0.0, 0.0}, {1.0, 0.0, 0.0});


    // remove cube
    EmbreeCubePtr cube = std::dynamic_pointer_cast<EmbreeCube>(scene->remove(cube_id));
    scene->commit();

    printRaycast(scene, {5.0, 0.0, 0.0}, {1.0, 0.0, 0.0});

    // add cube again
    scene->add(cube);
    scene->commit();

    printRaycast(scene, {5.0, 0.0, 0.0}, {1.0, 0.0, 0.0});

    // move cube
    
    auto T = cube->transform();
    T.t.x += 0.5;
    cube->setTransform(T);
    cube->apply();
    cube->markAsChanged();
    cube->commit();

    scene->commit();

    printRaycast(scene, {5.0, 0.0, 0.0}, {1.0, 0.0, 0.0});

    // add ground plane
    EmbreePlanePtr plane = std::make_shared<EmbreePlane>();
    plane->name = "Ground Plane";
    plane->setScale({1000.0, 1000.0, 1.0});
    plane->apply();
    plane->commit();

    scene->add(plane);
    scene->commit();

    printRaycast(scene, {-5.0, 0.0, 5.0}, {0.0, 0.0, -1.0});


} 

int main(int argc, char** argv)
{
    std::cout << "Rmagine Embree Scene Building" << std::endl;
    
    // std::cout << "SCENE EXAMPLE 1" << std::endl;
    // scene_1();

    // std::cout << "SCENE EXAMPLE 2" << std::endl;
    // scene_2();

    // std::cout << "SCENE EXAMPLE 3" << std::endl;
    // scene_3();

    // std::cout << "SCENE EXAMPLE 4" << std::endl;
    // scene_4();

    // std::cout << "SCENE EXAMPLE 5" << std::endl;
    // scene_5();

    // std::cout << "SCENE EXAMPLE 6" << std::endl;
    // scene_6();

    // std::cout << "SCENE EXAMPLE 7" << std::endl;
    // scene_7();

    std::cout << "SCENE EXAMPLE 8" << std::endl;
    scene_8();

    return 0;
}
