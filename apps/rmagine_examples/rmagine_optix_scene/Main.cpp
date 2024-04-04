#include <iostream>
#include <sstream>

// General rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/synthetic.h>

// #include <rmagine/map/OptixMap.hpp>
#include <rmagine/map/optix/OptixMesh.hpp>
#include <rmagine/map/optix/OptixInst.hpp>
#include <rmagine/map/optix/OptixScene.hpp>

#include <rmagine/util/prints.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/IDGen.hpp>

#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/map/optix/optix_shapes.h>

#include "mesh_changer.h"

using namespace rmagine;
namespace rm = rmagine;

SphericalModel single_ray_model()
{
    SphericalModel model;
    model.theta.min = 0.0;
    model.theta.inc = 0.0;
    model.theta.size = 1;

    model.phi.min = 0.0;
    model.phi.inc = 1.0;
    model.phi.size = 1;
    
    model.range.min = 0.0;
    model.range.max = 130.0;
    return model;
}

SphereSimulatorOptixPtr make_sim(OptixScenePtr scene)
{
    SphereSimulatorOptixPtr gpu_sim = std::make_shared<SphereSimulatorOptix>(scene);
    SphericalModel model = single_ray_model();
    gpu_sim->setModel(model);
    return gpu_sim;
}

void printRaycast(SphereSimulatorOptixPtr gpu_sim,
    Vector3 pos, 
    EulerAngles angles)
{
    Transform T = Transform::Identity();
    T.t = pos;
    T.R.set(angles);
    
    Memory<Transform, RAM> Tbm(1);
    Tbm[0] = T;

    Memory<Transform, VRAM_CUDA> Tbm_gpu;
    Tbm_gpu = Tbm;

    // std::cout << "Simulate!" << std::endl;

    using ResultT = Bundle<
        Ranges<VRAM_CUDA>,
        // Normals<VRAM_CUDA>,
        FaceIds<VRAM_CUDA>,
        GeomIds<VRAM_CUDA>,
        ObjectIds<VRAM_CUDA>
    >;

    // using ResultT = IntAttrAll<VRAM_CUDA>;

    ResultT res = gpu_sim->simulate<ResultT>(Tbm_gpu);

    // Download results
    Memory<float, RAM> ranges = res.ranges;
    // Memory<Vector, RAM> normals = res.normals;
    Memory<unsigned int, RAM> face_ids = res.face_ids;
    Memory<unsigned int, RAM> geom_ids = res.geom_ids;
    Memory<unsigned int, RAM> obj_ids = res.object_ids;

    // print results
    std::cout << "Result:" << std::endl;
    std::cout << "- range: " << ranges[0] << std::endl;
    // std::cout << "- normal: " << normals[0] << std::endl;
    std::cout << "- face id: " << face_ids[0] << std::endl;
    std::cout << "- geom id: " << geom_ids[0] << std::endl;
    std::cout << "- obj id: " << obj_ids[0] << std::endl;
}

void printRaycast(
    OptixScenePtr scene, 
    Vector3 pos, 
    EulerAngles angles)
{
    auto gpu_sim = make_sim(scene);
    printRaycast(gpu_sim, pos, angles);
}

OptixMeshPtr custom_mesh()
{
    OptixMeshPtr mesh = std::make_shared<OptixMesh>();

    Memory<Point, RAM> vertices_cpu(3);
    vertices_cpu[0] = {0.0, 0.5, 0.5};
    vertices_cpu[1] = {0.0, 0.5, -0.5};
    vertices_cpu[2] = {0.0, -0.5, -0.5};
    mesh->vertices = vertices_cpu;

    Memory<Face, RAM> faces_cpu(1);
    faces_cpu[0] = {0, 1, 2};
    mesh->faces = faces_cpu;

    mesh->computeFaceNormals();

    Memory<Vector, RAM> normals = mesh->face_normals;
    std::cout << "Computed normal: " << normals[0] << std::endl;

    Transform T;
    T.setIdentity();
    mesh->setTransform(T);
    mesh->apply();

    return mesh;
}

void scene_1()
{
    std::cout << "Make Optix Mesh" << std::endl;

    OptixScenePtr scene = std::make_shared<OptixScene>(); 

    OptixGeometryPtr geom1 = custom_mesh();
    geom1->commit();
    scene->add(geom1);


    OptixGeometryPtr geom2 = std::make_shared<OptixSphere>(50, 50);
    Transform T = Transform::Identity();
    T.t.y = 5.0;
    geom2->setTransform(T);
    geom2->apply();
    geom2->commit();
    scene->add(geom2);

    scene->commit();
    std::cout << "Scene committed" << std::endl;

    printRaycast(scene, {-5.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
    printRaycast(scene, {-5.0, 5.0, 0.0}, {0.0, 0.0, 0.0});
}

void scene_2()
{
    OptixScenePtr scene = std::make_shared<OptixScene>();

    OptixGeometryPtr geom = std::make_shared<OptixCube>();
    geom->name = "Cube";
    geom->commit();
    // scene->add(geom);

    OptixScenePtr geom_scene = geom->makeScene();
    geom_scene->commit();

    for(size_t i=0; i<10; i++)
    {
        OptixInstPtr inst_geom = geom_scene->instantiate();
        inst_geom->name = "Instance " + i;
        Transform T = Transform::Identity();
        T.t.y = static_cast<float>(i) * 2.0;
        T.t.x = 10.0;
        inst_geom->setTransform(T);
        inst_geom->apply();
        inst_geom->commit();
        scene->add(inst_geom);
    }
    
    scene->commit();
    auto sim = make_sim(scene);

    


    printRaycast(sim, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
    printRaycast(sim, {0.0, 18.0, 0.0}, {0.0, 0.0, 0.0});
    // printRaycast(sim, {0.0, 20.0, 0.0}, {0.0, 0.0, 0.0});


    OptixScenePtr geom_scene2 = std::make_shared<OptixScene>();
    geom_scene2->add(geom);
    OptixGeometryPtr geom2 = std::make_shared<OptixCube>();
    {
        Transform T = Transform::Identity();
        T.t.z = 5.0;
        geom2->setTransform(T);
        geom2->apply();
        geom2->name = "Cube2";
        geom2->commit();
    }
    geom_scene2->add(geom2);
    geom_scene2->commit();
    

    // add another
    OptixInstPtr inst_geom = geom_scene2->instantiate();
    {
        inst_geom->name = "Instance NEW";
        Transform T = Transform::Identity();
        T.t.z = 5.0;
        T.t.x = 10.0;
        inst_geom->setTransform(T);
        inst_geom->apply();
        inst_geom->commit();
    }
    scene->add(inst_geom);
    scene->commit();

    printRaycast(sim, {0.0, 0.0, 5.0}, {0.0, 0.0, 0.0});
    printRaycast(sim, {0.0, 0.0, 10.0}, {0.0, 0.0, 0.0});


}

void scene_3()
{
    // OptixScenePtr scene = std::make_shared<OptixScene>();

    // OptixGeometryPtr geom = std::make_shared<OptixCube>();
    // geom->name = "Cube";
    // geom->commit();
    // scene->add(geom);

    // OptixInstancesPtr insts = std::make_shared<OptixInstances>();

    // for(size_t i=0; i<10; i++)
    // {
    //     OptixInstPtr inst_geom = std::make_shared<OptixInst>();
    //     inst_geom->setGeometry(geom);
    //     inst_geom->name = "Instance " + i;
    //     Transform T = Transform::Identity();
    //     T.t.y = static_cast<float>(i);
    //     T.t.x = 10.0;
    //     inst_geom->setTransform(T);
    //     inst_geom->apply();
    //     insts->add(inst_geom);
    // }
    // insts->commit();

    // scene->setRoot(insts);
    // scene->commit();

    // printRaycast(scene, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

    // // add in front of 0
    // OptixInstPtr inst_geom = std::make_shared<OptixInst>();
    // {
    //     inst_geom->setGeometry(geom);
    //     Transform T = Transform::Identity();
    //     T.t.x = 5.0;
    //     inst_geom->setTransform(T);
    //     inst_geom->apply();
    // }
    // unsigned int inst_id;
    // inst_id = insts->add(inst_geom);
    // insts->commit();
    // scene->commit();
    // std::cout << "Added Instance with id " << inst_id << std::endl;

    // printRaycast(scene, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

    // // remove
    // insts->remove(inst_geom);
    // insts->commit();
    // scene->commit();
    // std::cout << "Removed Instance with id " << inst_id << std::endl;

    // printRaycast(scene, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

    
    // inst_id = insts->add(inst_geom);
    // insts->commit();
    // scene->commit();
    // std::cout << "Added Instance with id " << inst_id << std::endl;

    // printRaycast(scene, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

}

void scene_4()
{
    // sensor update problem

    // OptixScenePtr scene = std::make_shared<OptixScene>();

    // OptixGeometryPtr geom = std::make_shared<OptixCube>();
    // geom->name = "Cube";
    // geom->commit();
    // scene->add(geom);

    // OptixInstancesPtr insts = std::make_shared<OptixInstances>();

    // for(size_t i=0; i<2; i++)
    // {
    //     OptixInstPtr inst_geom = std::make_shared<OptixInst>();
    //     inst_geom->setGeometry(geom);
    //     inst_geom->name = "Instance " + i;
    //     Transform T = Transform::Identity();
    //     T.t.x = static_cast<float>(i) * 5.0 + 5.0;
    //     inst_geom->setTransform(T);
    //     inst_geom->apply();
    //     insts->add(inst_geom);
    // }
    // insts->commit();

    // scene->setRoot(insts);
    // scene->commit();

    // auto sim = make_sim(scene);
    // printRaycast(sim, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

    // OptixInstPtr inst_geom = std::make_shared<OptixInst>();
    // {
    //     inst_geom->setGeometry(geom);
    //     inst_geom->name = "Instance NEW";
    //     Transform T = Transform::Identity();
    //     T.t.x = 1.0;
    //     inst_geom->setTransform(T);
    //     inst_geom->apply();
    //     insts->add(inst_geom);
    // }
    // insts->commit();
    // scene->commit();

    // printRaycast(sim, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
}

void scene_5()
{
    // StopWatch sw;
    // double el;

    // OptixScenePtr scene = std::make_shared<OptixScene>();

    // // large sphere
    // OptixMeshPtr geom = std::make_shared<OptixSphere>(100, 100);
    // geom->name = "Sphere";
    // geom->commit();
    // scene->add(geom);
    // std::cout << "Constructed sphere with " << geom->faces.size() << " faces" << std::endl;

    // scene->setRoot(geom);
    // scene->commit();
    // std::cout << "Scene depth: " << scene->depth() << std::endl;

    // auto sim = make_sim(scene);
    // // shoot ray from x=-5.0 along x axis on unit cube
    // // -> range should be 4.5
    // printRaycast(sim, {-5.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

    // std::cout << "Changing scene geometry" << std::endl;
    // // move vertices 1m along x axis
    // // - this function calls a cuda kernel (zero copy)
    // sw();
    // moveVertices(geom->vertices, {1.0, 0.0, 0.0});
    // el = sw();
    // std::cout << "- move vertices: " << el * 1000.0 << "ms" << std::endl;

    // sw();
    // geom->computeFaceNormals();
    // geom->apply();
    // el = sw();
    // std::cout << "- postprocessing: " << el * 1000.0 << "ms" << std::endl;
    
    // sw();
    // geom->commit();
    // el = sw();
    // std::cout << "- geometry commit: " << el * 1000.0 << "ms" << std::endl;
    
    // sw();
    // scene->commit();
    // el = sw();

    // std::cout << "- scene commit: " << el * 1000.0 << "ms" << std::endl;

    // // shoot ray from x=-5.0 along x axis on unit cube
    // // -> range should be 5.5
    // printRaycast(sim, {-5.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
}

void scene_6()
{
    // OptixScenePtr scene = std::make_shared<OptixScene>();

    // OptixMeshPtr sphere1 = std::make_shared<OptixSphere>(30, 30);
    // sphere1->name = "Sphere Mesh 1";
    // sphere1->commit();

    // OptixMeshPtr sphere2 = std::make_shared<OptixSphere>(30, 30);
    // {
    //     Transform T = Transform::Identity();
    //     T.t.x = -1.0;
    //     sphere2->setTransform(T);
    //     sphere2->apply();
    // }
    // sphere2->commit();

    // // add as reference
    // scene->add(sphere1);
    // scene->add(sphere2);

    // OptixInstancesPtr insts = std::make_shared<OptixInstances>();

    ////////////////////////////////////
    // 1. add single mesh sphere (doesnt work in optix. needs an instance)
    // scene->add(geom1);

    ////////////////////////////////////
    // 2. add sphere as instance
    // {
    //     OptixInstPtr inst = std::make_shared<OptixInst>();
    //     inst->setGeometry(sphere1);

    //     Transform T = Transform::Identity();
    //     T.t.y = 0.0;
    //     inst->setTransform(T);
    //     inst->name = "Instance 1 Sphere 1";
    //     inst->apply();
    //     insts->add(inst);
    // }

    ////////////////////////////////////
    // 3. add two sphere as one instance
    // {
    //     OptixInstPtr inst = std::make_shared<OptixInst>();
    //     {
    //         OptixInstancesPtr insts_nested = std::make_shared<OptixInstances>();
    //         {
    //             OptixInstPtr inst2 = std::make_shared<OptixInst>();
    //             inst2->setGeometry(sphere2);
    //             inst2->apply();
    //             insts_nested->add(inst2);
                
    //             OptixInstPtr inst1 = std::make_shared<OptixInst>();
    //             inst1->setGeometry(sphere1);
    //             inst1->apply();
    //             insts_nested->add(inst1);

                
    //         }
    //         insts_nested->apply();
    //         insts_nested->commit();

    //         inst->setGeometry(insts_nested);
    //     }
    //     Transform T = Transform::Identity();
    //     T.t.y = 4.0;
    //     inst->setTransform(T);
    //     inst->apply();

    //     inst->name = "Instance 2, Mesh 1 and 2";
    //     insts->add(inst);
    // }

    // insts->apply();
    // insts->commit();

    // scene->setRoot(insts);
    // scene->commit();


    // // hit 2. mesh instanciated
    // printRaycast(scene, {-5.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

    // // hit 3. two meshes as one instance

    // std::cout << "-----------------" << std::endl;

    // printRaycast(scene, {-5.0, 4.0, 0.0}, {0.0, 0.0, 0.0});
    // printRaycast(scene, {0.0, 4.0, 0.0}, {0.0, 0.0, 0.0});


    




    
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Optix Scene Building" << std::endl;

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
        default: break;
    }

    return 0;
}
