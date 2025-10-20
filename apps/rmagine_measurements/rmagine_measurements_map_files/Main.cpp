// Core rmagine includes
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>
#include <rmagine/util/synthetic.h>
#include <rmagine/math/assimp_conversions.h>




using namespace rmagine;



size_t num_maps = 10;
size_t map_param = 100;

int main(int argc, char** argv)
{
    std::vector<double> results;
    for(size_t i = 1; i <= num_maps; i++)
    {
        aiScene scene = genSphere(map_param*i, map_param*i);

        //TODO: write to file in dat directory
    }

    std::cout << "\nFinished." << std::endl;

    return EXIT_SUCCESS;
}
