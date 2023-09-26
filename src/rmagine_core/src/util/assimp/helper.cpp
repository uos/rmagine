#include "rmagine/util/assimp/helper.h"
#include <rmagine/math/assimp_conversions.h>

#include <vector>
#include <algorithm>

namespace rmagine
{

void get_nodes_with_meshes(const aiNode* node, std::vector<const aiNode*>& mesh_nodes)
{
    if(node->mNumMeshes > 0)
    {
        mesh_nodes.push_back(node);
    }

    if(node->mNumChildren > 0)
    {
        // is parent. check if it has meshes anyway
        for(size_t i=0; i<node->mNumChildren; i++)
        {
            get_nodes_with_meshes(node->mChildren[i], mesh_nodes);
        }
    }
}

std::vector<const aiNode*> get_nodes_with_meshes(const aiNode* node)
{
    std::vector<const aiNode*> ret;
    get_nodes_with_meshes(node, ret);
    return ret;
}

std::vector<std::string> path_names(const aiNode* node)
{
    std::vector<std::string> res;

    const aiNode* it = node;
    while(it)
    {
        res.push_back(std::string(it->mName.C_Str()));
        it = it->mParent;
    }

    std::reverse(res.begin(), res.end());
    return res;
}

Matrix4x4 global_transform(const aiNode* node)
{
    Matrix4x4 M;
    M.setIdentity();

    const aiNode* it = node;
    while(it != NULL)
    {
        M = convert(it->mTransformation) * M;
        it = it->mParent;
    }

    return M;
}

} // namespace rmagine