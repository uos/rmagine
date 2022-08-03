#include <iostream>
#include <sstream>

// General rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/synthetic.h>

// #include <rmagine/map/OptixMap.hpp>
#include <rmagine/map/optix/OptixMesh.hpp>
#include <rmagine/map/optix/OptixInstance.hpp>

#include <rmagine/util/prints.h>
#include <rmagine/util/StopWatch.hpp>

using namespace rmagine;
namespace rm = rmagine;

void scene_1()
{
    std::cout << "Make Optix Mesh" << std::endl;

    OptixMeshPtr mesh = std::make_shared<OptixMesh>();


    { // FILL MESH
        std::cout << "Fill Buffers" << std::endl;
        Memory<Point, RAM> vertices_cpu(3);
        vertices_cpu[0] = {1.0, 0.0, 0.0};
        vertices_cpu[1] = {0.0, 1.0, 0.0};
        vertices_cpu[2] = {1.0, 1.0, 0.0};
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

    // MAKE INSTANCE
    OptixInstancePtr mesh_inst = std::make_shared<rm::OptixInstance>(mesh);




    std::cout << "done." << std::endl;
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
