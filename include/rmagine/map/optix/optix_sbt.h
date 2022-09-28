#ifndef RMAGINE_MAP_OPTIX_SBT_H
#define RMAGINE_MAP_OPTIX_SBT_H

#include "optix_definitions.h"
#include <rmagine/math/types.h>

namespace rmagine
{

union OptixGeomSBT;
struct OptixSceneSBT;
struct OptixMeshSBT;
struct OptixInstanceSBT;

struct OptixMeshSBT
{
    Vector* vertex_normals = nullptr;
    Vector* face_normals = nullptr;
    unsigned int id = 0;
};

struct OptixInstanceSBT
{
    OptixSceneSBT* scene = nullptr;
};

union OptixGeomSBT
{
    OptixMeshSBT mesh_data;
    OptixInstanceSBT inst_data;
};

struct OptixSceneSBT
{
    OptixSceneType type;
    unsigned int n_geometries = 0;
    OptixGeomSBT* geometries = nullptr;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SBT_H