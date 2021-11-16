#include "imagine/map/AssimpMap.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

namespace imagine {

AssimpMap::AssimpMap(std::string filename)
{
    Assimp::Importer importer;
    scene = importer.ReadFile( filename, 
        aiProcess_CalcTangentSpace       | 
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType);

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