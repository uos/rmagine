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
 * @brief Assimp Conversions
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MATH_ASSIMP_CONVERSIONS_H
#define RMAGINE_MATH_ASSIMP_CONVERSIONS_H

#include "types.h"
#include <assimp/matrix4x4.h>
#include <assimp/matrix3x3.h>
#include <assimp/vector3.h>
#include <assimp/aabb.h>

namespace rmagine 
{

// DEFINITIONS
inline Vector3 convert(const aiVector3D& av);
inline void convert(const aiVector3D& av, Vector3& v);

inline Matrix3x3 convert(const aiMatrix3x3& aT);
inline void convert(const aiMatrix3x3& aT, Matrix3x3& T);

inline Matrix4x4 convert(const aiMatrix4x4& aT);
inline void convert(const aiMatrix4x4& aT, Matrix4x4& T);

inline AABB convert(const aiAABB& aaabb);
inline void convert(const aiAABB& aaabb, AABB& aabb);

// IMPLEMENTATIONS
inline Vector3 convert(const aiVector3D& av)
{
    return {av.x, av.y, av.z};
}

inline void convert(const aiVector3D& av, Vector3& v)
{
    v.x = av.x;
    v.y = av.y;
    v.z = av.z;
}

inline Matrix3x3 convert(const aiMatrix3x3& aT)
{
    Matrix3x3 T;
    convert(aT, T);
    return T;
}

inline void convert(const aiMatrix3x3& aT, Matrix3x3& T)
{
    T(0,0) = aT.a1;
    T(0,1) = aT.a2;
    T(0,2) = aT.a3;
    T(1,0) = aT.b1;
    T(1,1) = aT.b2;
    T(1,2) = aT.b3;
    T(2,0) = aT.c1;
    T(2,1) = aT.c2;
    T(2,2) = aT.c3;
}

inline Matrix4x4 convert(const aiMatrix4x4& aT)
{
    Matrix4x4 T;
    convert(aT, T);
    return T;
}

inline void convert(const aiMatrix4x4& aT, Matrix4x4& T)
{
    T(0,0) = aT.a1;
    T(0,1) = aT.a2;
    T(0,2) = aT.a3;
    T(0,3) = aT.a4;
    T(1,0) = aT.b1;
    T(1,1) = aT.b2;
    T(1,2) = aT.b3;
    T(1,3) = aT.b4;
    T(2,0) = aT.c1;
    T(2,1) = aT.c2;
    T(2,2) = aT.c3;
    T(2,3) = aT.c4;
    T(3,0) = aT.d1;
    T(3,1) = aT.d2;
    T(3,2) = aT.d3;
    T(3,3) = aT.d4;
}

inline AABB convert(const aiAABB& aaabb)
{
    return {convert(aaabb.mMin), convert(aaabb.mMax)};
}

inline void convert(const aiAABB& aaabb, AABB& aabb)
{
    aabb.min = convert(aaabb.mMin);
    aabb.max = convert(aaabb.mMax);
}

} // namespace rmagine 

#endif // RMAGINE_MATH_ASSIMP_CONVERSIONS_H