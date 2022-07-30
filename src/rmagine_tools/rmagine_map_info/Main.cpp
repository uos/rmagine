
#include <rmagine/util/assimp/prints.h>
#include <assimp/Importer.hpp>

using namespace rmagine;

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

    Assimp::Importer importer;
    const aiScene* ascene = importer.ReadFile(filename, 0);

    print(ascene);

    return 0;
}
