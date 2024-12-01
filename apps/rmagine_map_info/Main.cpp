
#include <rmagine/util/assimp/prints.h>
#include <assimp/Importer.hpp>
#include <rmagine/map/AssimpIO.hpp>

namespace rm = rmagine;

int main(int argc, char** argv)
{
    std::cout << "Rmagine Map Info" << std::endl;

    // minimum 1 argument
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " mesh_file" << std::endl;
        return 0;
    }

    std::string filename = argv[1];

    std::cout << "Inputs: " << std::endl;
    std::cout << "- filename: " << filename << std::endl;

    rm::AssimpIO io;
    const aiScene* ascene = io.ReadFile(filename, 0);

    if(ascene)
    {
        rm::print(ascene);
    } else {
        std::cout << "Loading failed" << std::endl;
        std::cerr << io.Importer::GetErrorString() << std::endl;
    }

    return 0;
}
