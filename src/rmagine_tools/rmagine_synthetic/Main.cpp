#include <iostream>

// General rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/synthetic.h>

#include <assimp/Exporter.hpp>

#include <unordered_map>
#include <functional>


#include <iomanip>

using namespace rmagine;


int main(int argc, char** argv)
{
    std::cout << "Rmagine Synthetic" << std::endl;


    // minimum 2 arguments
    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " mesh_type mesh_file" << std::endl;
        std::cout << "- mesh_type: plane | cube | sphere | cylinder " << std::endl;

        return 0;
    }

    std::string mesh_type = argv[1];
    std::string filename = argv[2];
    

    std::cout << "Inputs: " << std::endl;
    std::cout << "- mesh_type: " << mesh_type << std::endl;
    std::cout << "- filename: " << filename << std::endl;



    std::unordered_map<std::string, std::function<aiScene()> > gen_map;

    gen_map["plane"] = []() { return genPlane(); };
    gen_map["cube"] = []() { return genCube(); };
    gen_map["sphere"] = []() { return genSphere(); };
    gen_map["cylinder"] = []() {return genCylinder(); };
    

    auto gen_map_it = gen_map.find(mesh_type);

    if(gen_map_it != gen_map.end())
    {
        aiScene scene = gen_map_it->second();

        Assimp::Exporter exporter;
        exporter.Export(&scene, "ply", filename);
    } else {
        std::cout << "mesh_type '" << mesh_type << "' not implemented." << std::endl; 
    }


    return 0;
}
