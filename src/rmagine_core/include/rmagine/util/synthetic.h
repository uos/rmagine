/*
 * Copyright (c) 2022, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * 
 * @brief Synthetic data generation
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_UTIL_SYNTHETIC_H
#define RMAGINE_UTIL_SYNTHETIC_H

#include <assimp/scene.h>
#include <vector>
#include <rmagine/types/mesh_types.h>
#include <rmagine/types/shared_functions.h>

namespace rmagine 
{

RMAGINE_API
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
RMAGINE_API
void genSphere(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int num_long,
    unsigned int num_lat
);

RMAGINE_API
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
RMAGINE_API
void genCube(
    std::vector<Vector3>& vertices, 
    std::vector<Face>& faces,
    unsigned int side_triangles_exp=1);

RMAGINE_API
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
RMAGINE_API
void genPlane(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int side_triangles_exp=1
);

RMAGINE_API
aiScene genPlane(unsigned int side_triangles_exp=1);

/**
 * @brief Generates Cylinder of 1m height and 1m diameter (0.5m radius)
 * 
 * @param vertices 
 * @param faces 
 * @param side_faces number of rectangles used to express the curvature
 */
RMAGINE_API
void genCylinder(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int side_faces = 100);

RMAGINE_API
aiScene genCylinder(unsigned int side_faces = 100);


} // namespace rmagine

#endif // RMAGINE_UTIL_SYNTHETIC_H