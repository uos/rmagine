#include "rmagine/map/AssimpMap.hpp"

namespace rmagine {

AssimpMap::AssimpMap(const aiScene* scene)
:scene(scene)
{

}

AssimpMap::AssimpMap(std::string filename)
{
    Assimp::Importer importer;
    scene = importer.ReadFile( filename, 0);

    if(!scene)
    {
        throw std::runtime_error("Assimp Could not load file");
    }

    if(!scene->HasMeshes())
    {
        throw std::runtime_error("ERROR: file contains no meshes.");
    }
}

} // namespace mamcl