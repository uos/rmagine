#ifndef RMAGINE_UTIL_ASSIMP_HELPER_H
#define RMAGINE_UTIL_ASSIMP_HELPER_H

#include <assimp/scene.h>
#include <vector>
#include <rmagine/math/types.h>
#include <rmagine/types/shared_functions.h>

namespace rmagine
{

void get_nodes_with_meshes(
    const aiNode* node, 
    std::vector<const aiNode*>& mesh_nodes);

std::vector<const aiNode*> get_nodes_with_meshes(
    const aiNode* node);

std::vector<std::string> path_names(const aiNode* node);

Matrix4x4 global_transform(const aiNode* node);

} // namespace rmagine

#endif // RMAGINE_UTIL_ASSIMP_HELPER_H