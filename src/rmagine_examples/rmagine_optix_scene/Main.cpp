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


void printRaycast(OptixGeometryPtr geom, Vector3 pos, EulerAngles angles)
{
    std::cout << "Create Sphere Simulator" << std::endl;
    SphereSimulatorOptixPtr gpu_sim = std::make_shared<SphereSimulatorOptix>(geom);


    std::cout << "Create single ray model" << std::endl;
    SphericalModel model = single_ray_model();
    gpu_sim->setModel(model);
    Transform T;
    T.setIdentity();
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

void scene_1()
{
    std::cout << "Make Optix Mesh" << std::endl;

    OptixScenePtr scene = std::make_shared<OptixScene>(); 


    OptixMeshPtr mesh = std::make_shared<OptixMesh>();


    { // FILL MESH
        std::cout << "Fill Buffers" << std::endl;
        Memory<Point, RAM> vertices_cpu(3);
        vertices_cpu[0] = {0.0, 0.5, 0.5};
        vertices_cpu[1] = {0.0, 0.5, -0.5};
        vertices_cpu[2] = {0.0, -0.5, -0.5};
        mesh->vertices = vertices_cpu;
        std::cout << "- vertices" << std::endl;

        Memory<Face, RAM> faces_cpu(1);
        faces_cpu[0] = {0, 1, 2};
        mesh->faces = faces_cpu;
        std::cout << "- faces" << std::endl;

        Transform T;
        T.setIdentity();
        mesh->setTransform(T);
        std::cout << "- transform" << std::endl; 

        Vector3 s = {1.0, 1.0, 1.0};
        mesh->setScale(s);
        std::cout << "- scale" << std::endl;

        mesh->apply();
        mesh->commit();

        // TODO
        // mesh->computeFaceNormals();
    }

    OptixInstancesPtr insts = std::make_shared<OptixInstances>();

    // MAKE INSTANCE
    for(size_t i=0; i<100; i++)
    {
        OptixInstPtr mesh_inst = std::make_shared<OptixInst>(mesh);

        Transform T;
        T.setIdentity();
        T.t.x = static_cast<float>(i);
        mesh_inst->setTransform(T);
        mesh_inst->apply();

        unsigned int id = insts->add(mesh_inst);
        std::cout << "Created instance " << id << std::endl;
    }

    std::cout << "Commit Instances" << std::endl;
    insts->commit();

    printRaycast(insts, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

    // push forward
    for(auto elem : insts->instances())
    {
        OptixInstPtr inst = elem.second;
        Transform T = inst->transform();
        T.t.x += 5.0;
        inst->setTransform(T);
        inst->apply();
    }

    StopWatch sw;
    double el;


    sw();
    insts->commit();
    el = sw();
    std::cout << "Updated instances in " << el << "s" << std::endl;

    printRaycast(insts, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});



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
