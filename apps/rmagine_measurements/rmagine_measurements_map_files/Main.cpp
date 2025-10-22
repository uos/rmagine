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

        return 0;
    }

    AssimpIO io;
    for(size_t i = 1; i <= num_maps; i++)
    {
        unsigned int num_lon_and_lat = static_cast<unsigned int>(static_cast<double>(map_param)*sqrt(static_cast<double>(i)));
        aiScene scene = genSphere(num_lon_and_lat, num_lon_and_lat);

        //TODO: write to file in dat directory
        std::string filename = "";// argv[1] ...;

        io.Export(&scene, "ply", filename);
    }

    std::cout << "\nFinished." << std::endl;

    return EXIT_SUCCESS;
}
