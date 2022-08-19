#include <iostream>
#include <sstream>

// General rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/synthetic.h>

#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/map/embree/EmbreePoints.hpp>

#include <rmagine/map/embree/embree_shapes.h>
#include <rmagine/util/prints.h>
#include <rmagine/util/StopWatch.hpp>



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

    if(sphere->parents.find(scene) == sphere->parents.end())
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
    std::cout << sphere->parents.count(scene) << std::endl;
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
        unsigned int objID;
        std::string type;
        if(rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID)
        {
            objID = rayhit.hit.instID[0];
            type = "instance";
        } else {
            objID = rayhit.hit.geomID;
            type = "geometry";
        }

        auto geom = scene->get(objID);

        std::cout << "- type: " << type << std::endl;
        std::cout << "- name: " << geom->name << std::endl;
        std::cout << "- objID: " << objID << std::endl;
        std::cout << "- geomID: " << rayhit.hit.geomID << std::endl;
        std::cout << "- faceID: " << rayhit.hit.primID << std::endl;
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
    
    EmbreeScenePtr sphere_scene = sphere->makeScene();
    sphere_scene->commit();

    // make N sphere instances
    int Ninstances = 100;
    for(int i=0; i < Ninstances; i++)
    {
        EmbreeInstancePtr sphere_inst = sphere_scene->instantiate();

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

        EmbreeInstancePtr sphere_inst = sphere_scene->instantiate();

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

EmbreeScenePtr make_scene()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();
    scene->setQuality(RTC_BUILD_QUALITY_LOW);
    scene->setFlags(RTC_SCENE_FLAG_DYNAMIC);

    { 
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

            Transform T;
            T.setIdentity();
            T.t.y = static_cast<float>(i - Ninstances / 2);

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
    }

    { // NOT INSTANCES SPHERE
        int Ninstances = 100;
        for(int i=0; i<Ninstances; i++)
        {
            EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>();
            std::stringstream ss;
            ss << "Sphere Mesh " << i;
            sphere->name = ss.str();

            Transform T;
            T.setIdentity();
            T.t.x = -10.0;
            T.t.y = static_cast<float>(i - (Ninstances / 2) );

            Vector3 scale = {
                0.01f * static_cast<float>(i + 1),
                0.01f * static_cast<float>(i + 1),
                0.01f * static_cast<float>(i + 1)
            };

            sphere->setTransform(T);
            sphere->setScale(scale);
            sphere->apply();
            sphere->commit();

            scene->add(sphere);
        }
        
    }

    { // CUBE MESH
        EmbreeCubePtr cube = std::make_shared<EmbreeCube>();
        cube->name = "Cube";

        Transform T;
        T.setIdentity();
        T.t.x += 10.0;
        T.t.z += 0.5;
        cube->setTransform(T);
        cube->apply();
        cube->commit();

        scene->add(cube);
    }

    return scene;
}

void scene_9()
{
    StopWatch sw;
    double el;


    EmbreeScenePtr scene = make_scene();

    // test scene update runtimes

    // 1. no change update

    sw();
    scene->commit();
    el = sw();
    std::cout << "No changes update in " << el << "s" << std::endl;

    // should hit "Sphere Standalone" here
    // printRaycast(scene, {5.0, 10.0, 0.5}, {1.0, 0.0, 0.0});

    
    // 2. instances update
    {
        
        sw();
        for(size_t i=0; i<100; i++)
        {
            EmbreeInstancePtr inst = scene->getAs<EmbreeInstance>(i);
            auto T = inst->transform();
            T.t.x -= 5.0;
            inst->setTransform(T);
            inst->apply();
            inst->commit();
        }
        double prep_el = sw();

        sw();
        scene->commit();
        el = sw();
        std::cout << "Move " << 100 << " instances and update scene in " << prep_el + el << "s (" << prep_el << " + " << el << ")" << std::endl;

        printRaycast(scene, {-5.0, 49.0, 0.0}, {1.0, 0.0, 0.0});
    }

    // 3. mesh update
    {
        sw();
        for(size_t i=100; i<200; i++)
        {
            EmbreeMeshPtr sphere = scene->getAs<EmbreeMesh>(i);
            auto T = sphere->transform();
            T.t.x -= 5.0;
            sphere->setTransform(T);
            sphere->apply();
            sphere->commit();
        }

        double prep_el = sw();

        sw();
        scene->commit();
        el = sw();
        std::cout << "Move " << 100 << " meshes and update scene in " << prep_el + el << "s (" << prep_el << " + " << el << ")" << std::endl;

        printRaycast(scene, {-15.0, 49.0, 0.0}, {1.0, 0.0, 0.0});
    }
}

void scene_10()
{
    EmbreeScenePtr scene = make_scene();
    EmbreeScenePtr scene2 = make_scene();

    scene->commit();

    EmbreeGeometryPtr inst99 = scene2->get(99);
    auto T = inst99->transform();
    T.t.x += 5.0;
    inst99->setTransform(T);
    inst99->apply();
    inst99->commit();

    scene2->commit();

    printRaycast(scene, {-5.0, 49.0, 0.0}, {1.0, 0.0, 0.0});

    printRaycast(scene2, {-5.0, 49.0, 0.0}, {1.0, 0.0, 0.0});
}

void scene_11()
{
    EmbreeScenePtr scene = make_scene();
    EmbreeScenePtr scene2 = std::make_shared<EmbreeScene>();


    EmbreeCylinderPtr cylinder = std::make_shared<EmbreeCylinder>();
    
    Transform T;
    T.setIdentity();
    T.t.z = 50.0;
    cylinder->setTransform(T);
    cylinder->apply();
    cylinder->commit();
    cylinder->name = "MyCylinder";
    scene2->add(cylinder);
    scene2->commit();
    
    printRaycast(scene2, {0.0, 0.0, 50.0}, {1.0, 0.0, 0.0});

    scene->add(cylinder);
    // scene->integrate(scene2);
    scene->commit();

    

    // now cylinder is attached to two scenes
    std::cout << "SCENE -> ID" << std::endl;
    auto ids = cylinder->ids();
    for(auto elem : ids)
    {
        
        if(elem.first.lock() == scene)
        {
            std::cout << "- scene 1";
        } else if(elem.first.lock() == scene2) {
            std::cout << "- scene 2";
        }
        std::cout << " -> " << elem.second << std::endl;
    }


    printRaycast(scene, {0.0, 0.0, 50.0}, {1.0, 0.0, 0.0});
    printRaycast(scene2, {0.0, 0.0, 50.0}, {1.0, 0.0, 0.0});




    // remove cylinder from scene1

    scene->remove(cylinder);
    scene->commit();

    std::cout << "Cylinder removed from scene 1" << std::endl;

    std::cout << "SCENE -> ID" << std::endl;
    ids = cylinder->ids();
    for(auto elem : ids)
    {
        
        if(elem.first.lock() == scene)
        {
            std::cout << "- scene 1";
        } else if(elem.first.lock() == scene2) {
            std::cout << "- scene 2";
        }
        std::cout << " -> " << elem.second << std::endl;
    }
}

void scene_12()
{
    std::cout << "Create a scene" << std::endl;
    EmbreeScenePtr scene = make_scene();
    scene->commit();



    std::cout << "scene graph: " << scene->geometries().size() << std::endl;
    std::cout << "-- meshes: " << scene->count<EmbreeMesh>() << std::endl;
    std::cout << "-- instances: " << scene->count<EmbreeInstance>() << std::endl;

    std::unordered_set<EmbreeGeometryPtr> leaf_geometries = scene->findLeafs();

    std::cout << "Total of leaf geometries: " << leaf_geometries.size() << std::endl; 
}

void scene_13()
{
    std::cout << "Create a scene" << std::endl;
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();
    scene->setQuality(RTC_BUILD_QUALITY_LOW);
    scene->setFlags(RTC_SCENE_FLAG_DYNAMIC);

    // add cube
    EmbreeMeshPtr cube = std::make_shared<EmbreeCube>();
    cube->name = "MyCube";
    cube->commit();
    scene->add(cube);
    

    scene->commit();

    printRaycast(scene, {-5.0, 0.0, 0.0}, {1.0, 0.0, 0.0});

    // change cube vertices: shift 1 along x axis
    for(size_t i=0; i<cube->vertices.size(); i++)
    {
        cube->vertices[i].x += 1.0;
    }
    // apply changes
    cube->apply();
    // commit geometry
    cube->commit();

    // commit scene
    scene->commit();

    printRaycast(scene, {-5.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
}

void scene_14()
{
    std::cout << "Create a scene" << std::endl;
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();
    scene->setQuality(RTC_BUILD_QUALITY_LOW);
    scene->setFlags(RTC_SCENE_FLAG_DYNAMIC);


    // 0. define two geometries to spawn
    EmbreeMeshPtr sphere1 = std::make_shared<EmbreeSphere>();
    sphere1->name = "1. sphere mesh";
    sphere1->commit();

    EmbreeMeshPtr sphere2 = std::make_shared<EmbreeSphere>();
    {
        Transform T = Transform::Identity();
        T.t.x = -1.0;
        sphere2->setTransform(T);
        sphere2->apply();
    }
    sphere2->commit();

    ////////////////////////////////////
    // 1. add single mesh sphere
    scene->add(sphere1);

    ////////////////////////////////////
    // 2. add sphere as instance
    EmbreeInstancePtr sphere_inst = sphere1->instantiate();
    {
        Transform T = Transform::Identity();
        T.t.y = 2.0;
        sphere_inst->setTransform(T);
        sphere_inst->apply();
        sphere_inst->commit();
        sphere_inst->name = "2. single sphere instance";
    }
    scene->add(sphere_inst);

    ///////////////////////////////////
    // 3. add two spheres as one instance
    EmbreeScenePtr two_sphere_scene = std::make_shared<EmbreeScene>();
    {
        two_sphere_scene->add(sphere1);
        two_sphere_scene->add(sphere2);
    }
    two_sphere_scene->commit();
    
    EmbreeInstancePtr two_sphere_inst = two_sphere_scene->instantiate();
    {
        Transform T = Transform::Identity();
        T.t.y = 4.0;
        two_sphere_inst->setTransform(T);
        two_sphere_inst->apply();
        two_sphere_inst->name = "3. two spheres instance";
    }
    two_sphere_inst->commit();

    scene->add(two_sphere_inst);
    scene->commit();

    // hit 1. mesh only 
    printRaycast(scene, {-5.0, 0.0, 0.0}, {1.0, 0.0, 0.0});

    // hit 2. mesh instanciated
    printRaycast(scene, {-5.0, 2.0, 0.0}, {1.0, 0.0, 0.0});

    // hit 3. two meshes as one instance
    printRaycast(scene, {-5.0, 4.0, 0.0}, {1.0, 0.0, 0.0});

}

/**
 * @brief Build a scene with pointcloud
 * 
 */
void scene_15()
{
    // Create scene with pointcloud as EmbreePoints type
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();
    scene->setQuality(RTC_BUILD_QUALITY_LOW);
    scene->setFlags(RTC_SCENE_FLAG_DYNAMIC);

    EmbreePointsPtr points = std::make_shared<EmbreePoints>(100 * 100);

    // fill
    for(size_t i=0; i<100; i++)
    {
        for(size_t j=0; j<100; j++)
        {
            PointWithRadius p;
            p.p = {
                0.0,
                static_cast<float>(i) - 50.0f,
                static_cast<float>(j) - 50.0f
            };
            p.r = 0.1;
            points->points[i * 100 + j] = p;
        }
    }

    points->apply();
    points->commit();

    scene->add(points);
    scene->commit();

    printRaycast(scene, {-5.0, 0.0, 0.0}, {1.0, 0.0, 0.0});
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Embree Scene Building" << std::endl;

    int example = 1;

    if(argc > 1)
    {
        example = std::stoi( argv[1] );
    }

    std::cout << "SCENE EXAMPLE " << example << std::endl;
    
    switch(example)
    {
        case 1: scene_1(); break;
        case 2: scene_2(); break;
        case 3: scene_3(); break;
        case 4: scene_4(); break;
        case 5: scene_5(); break;
        case 6: scene_6(); break;
        case 7: scene_7(); break;
        case 8: scene_8(); break;
        case 9: scene_9(); break;
        case 10: scene_10(); break;
        case 11: scene_11(); break;
        case 12: scene_12(); break;
        case 13: scene_13(); break;
        case 14: scene_14(); break;
        case 15: scene_15(); break;
        default: break;
    }

    return 0;
}
