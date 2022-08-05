#include <iostream>
#include <sstream>

// General rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/synthetic.h>

// #include <rmagine/map/OptixMap.hpp>
#include <rmagine/map/optix/OptixMesh.hpp>
#include <rmagine/map/optix/OptixInst.hpp>
#include <rmagine/map/optix/OptixInstances.hpp>
#include <rmagine/map/optix/OptixScene.hpp>

#include <rmagine/util/prints.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/IDGen.hpp>

#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/map/optix/optix_shapes.h>

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


void printRaycast(
    OptixGeometryPtr geom, 
    Vector3 pos, 
    EulerAngles angles)
{
    std::cout << "Create Sphere Simulator" << std::endl;
    SphereSimulatorOptixPtr gpu_sim = std::make_shared<SphereSimulatorOptix>(geom);


    std::cout << "Create single ray model" << std::endl;
    SphericalModel model = single_ray_model();
    gpu_sim->setModel(model);
    Transform T = Transform::Identity();
    T.t = pos;
    T.R.set(angles);
    
    Memory<Transform, RAM> Tbm(1);
    Tbm[0] = T;

    Memory<Transform, VRAM_CUDA> Tbm_gpu;
    Tbm_gpu = Tbm;

    std::cout << "Simulate!" << std::endl;

    Memory<float, RAM> res = gpu_sim->simulateRanges(Tbm_gpu);
    std::cout << "- ranges: " << res.size() << std::endl;
    std::cout << "- range 0: " << res[0] << std::endl;
    std::cout << "done." << std::endl;
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

    OptixMeshPtr mesh1 = custom_mesh();
    mesh1->commit();
    OptixMeshPtr mesh2 = std::make_shared<OptixSphere>(50, 50);
    mesh2->commit();
    scene->add(mesh1);
    scene->add(mesh2);


    OptixInstancesPtr insts = std::make_shared<OptixInstances>();

    {   // two custom instances (5, 0, 0) and (5, 5, 0)
        OptixInstPtr mesh_inst_1 = std::make_shared<OptixInst>();
        mesh_inst_1->setGeometry(mesh1);

        Transform T = Transform::Identity();
        T.t.x = 5.0;
        mesh_inst_1->setTransform(T);
        mesh_inst_1->apply();
        insts->add(mesh_inst_1);
        std::cout << T << std::endl;

        OptixInstPtr mesh_inst_2 = std::make_shared<OptixInst>();
        mesh_inst_2->setGeometry(mesh1);
        T.t.y = 5.0;
        mesh_inst_2->setTransform(T);
        mesh_inst_2->apply();
        insts->add(mesh_inst_2);
    }

    { // 10 sphere instances at z = 10 from x=0 to x=10
        for(size_t i=0; i<10; i++)
        {
            OptixInstPtr mesh_inst = std::make_shared<OptixInst>();

            mesh_inst->setGeometry(mesh2);

            Transform T;
            T.setIdentity();
            T.t.z = 10.0;
            T.t.x = static_cast<float>(i);
            mesh_inst->setTransform(T);
            mesh_inst->apply();

            unsigned int id = insts->add(mesh_inst);
            std::cout << "Created instance " << id << std::endl;
        }
    }

    std::cout << "Commit Instances" << std::endl;
    insts->commit();

    scene->setRoot(insts);

    printRaycast(insts, {0.0, 0.0, 10.0}, {0.0, 0.0, 0.0});
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
        default: break;
    }

    return 0;
}
