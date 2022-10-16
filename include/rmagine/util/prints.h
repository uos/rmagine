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
 * @brief Prints for Rmagine math types
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_UTIL_PRINTS_H
#define RMAGINE_UTIL_PRINTS_H

#include <iostream>
#include <rmagine/math/types.h>

inline std::ostream& operator<<(std::ostream& os, const rmagine::Vector& v)
{
    os << "v[" << v.x << "," << v.y << "," << v.z << "]";

    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::AABB& aabb)
{
    os << "AABB [" << aabb.min <<  " - " << aabb.max << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::Quaternion& q)
{
    os << "q[" << q.x << "," << q.y << "," << q.z << "," << q.w << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::EulerAngles& e)
{
    os << "E[" << e.roll << ", " << e.pitch << ", " << e.yaw << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::Matrix3x3& M)
{
    os << "M3x3[\n";
    os << " " << M(0, 0) << " " << M(0, 1) << " " << M(0, 2) << "\n";
    os << " " << M(1, 0) << " " << M(1, 1) << " " << M(1, 2) << "\n";
    os << " " << M(2, 0) << " " << M(2, 1) << " " << M(2, 2) << "\n";
    os << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::Matrix4x4& M)
{
    os << "M4x4[\n";
    os << " " << M(0, 0) << " " << M(0, 1) << " " << M(0, 2) << " " << M(0, 3) << "\n";
    os << " " << M(1, 0) << " " << M(1, 1) << " " << M(1, 2) << " " << M(1, 3) << "\n";
    os << " " << M(2, 0) << " " << M(2, 1) << " " << M(2, 2) << " " << M(2, 3) << "\n";
    os << " " << M(3, 0) << " " << M(3, 1) << " " << M(3, 2) << " " << M(3, 3) << "\n";
    os << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::Transform& T)
{
    rmagine::EulerAngles e;
    e.set(T.R);
    os << "T[" << T.t << ", " << e << "]";
    return os;
}

#endif // RMAGINE_UTIL_PRINTS_H