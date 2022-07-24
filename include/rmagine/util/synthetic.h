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

/**
 * @brief Generates Sphere with diameter of 1
 * 
 * @param vertices 
 * @param faces 
 * @param num_long 
 * @param num_lat 
 */
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
 * Nt = 2 * 4^(a-1) 
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

/**
 * @brief Generates 1mx1m plane centered in (0,0,0) with normal (0,0,1)
 * 
 * Nt: Number of triangles per plane
 * a: side_triangles_exp
 * Nt = 2 * 4^(a-1)
 * 
 * a = 1 -> Nt = 2
 * a = 2 -> Nt = 8
 * a = 3 -> Nt = 32
 * 
 * @param vertices 
 * @param faces 
 * @param side_triangles_exp 
 */
void genPlane(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int side_triangles_exp=1
);

aiScene genPlane(unsigned int side_triangles_exp=1);


void genCylinder(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int side_faces = 100);


} // namespace rmagine

#endif // RMAGINE_UTIL_SYNTHETIC_H