#ifndef RMAGINE_UTIL_SYNTHETIC_H
#define RMAGINE_UTIL_SYNTHETIC_H

#include <assimp/scene.h>
#include <vector>
#include <rmagine/types/mesh_types.h>

namespace rmagine 
{

aiScene createAiScene(
    const std::vector<Vector3>& vertices,
    const std::vector<Face>& faces);


void genSphere(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int num_long,
    unsigned int num_lat
);

aiScene genSphere(unsigned int num_long = 50, unsigned int num_lat = 50);

/**
 * @brief Each side has per default two triangles. 
 * 
 * Nt: Number of triangles per side
 * a: side_triangles_exp
 * Nt = 2 * 4^a 
 * 
 * if a = 1 -> Nt = 2
 * if a = 2 -> Nt = 8
 * if a = 3 -> Nt = 32
 * 
 * total triangles can be computed by 12 * 4^(side_triangles_exp)
 * 
 * @param vertices 
 * @param faces 
 * @param side_triangles_exp
 * 
 * 
 *  
 */
void genCube(
    std::vector<Vector3>& vertices, 
    std::vector<Face>& faces,
    unsigned int side_triangles_exp=1);

aiScene genCube(unsigned int side_triangles_exp=1);

} // namespace rmagine

#endif // RMAGINE_UTIL_SYNTHETIC_H