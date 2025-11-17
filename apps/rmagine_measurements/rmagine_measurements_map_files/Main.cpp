#include <filesystem>

// Core rmagine includes
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>
#include <rmagine/util/synthetic.h>
#include <rmagine/math/assimp_conversions.h>




using namespace rmagine;



size_t num_maps = 20;
size_t map_param = 500;

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " path_to_directory_for_mesh_files" << std::endl;

        return EXIT_SUCCESS;
    }

    std::filesystem::path dir(argv[1]);
    if(!std::filesystem::exists(dir))
    {
        std::cout << "Directory [" << dir.c_str() << "] does not exist..." << std::endl;
        std::cout << "Usage: " << argv[0] << " path_to_directory_for_mesh_files" << std::endl;

        return EXIT_SUCCESS;
    }
    if(!std::filesystem::is_directory(dir))
    {
        std::cout << "Path [" << dir.c_str() << "] does not refer to a directory..." << std::endl;
        std::cout << "Usage: " << argv[0] << " path_to_directory_for_mesh_files" << std::endl;

        return EXIT_SUCCESS;
    }
    
    AssimpIO io;
    for(size_t i = 0; i <= num_maps; i++)
    {
        unsigned int num_lon_and_lat = static_cast<unsigned int>(static_cast<double>(map_param)*sqrt(static_cast<double>(i)));
        if(i == 0)
            num_lon_and_lat = 10;
        aiScene scene = genSphere(num_lon_and_lat, num_lon_and_lat);

        std::string filename = "sphere_";
        filename += std::to_string(i/2);
        filename += (i%2 == 0 ? "_0" : "_5");
        filename += "_million_faces.ply";

        std::filesystem::path file = dir;
        file.append(filename);

        std::cout << i << ": [" << file.c_str() << "]"<< std::endl;

        io.Export(&scene, "ply", file.c_str());
    }

    std::cout << "\nFinished." << std::endl;

    return EXIT_SUCCESS;
}
