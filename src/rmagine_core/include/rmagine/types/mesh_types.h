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
 * @brief Math type that are required for meshes
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */


#ifndef RMAGINE_TYPES_MESH_TYPES_H
#define RMAGINE_TYPES_MESH_TYPES_H

#include <rmagine/math/types.h>
#include <tuple>
#include <type_traits>

namespace rmagine
{

using Vertex = Point;

struct Face {
    unsigned int v0;
    unsigned int v1;
    unsigned int v2;

    // Other access functions
    // use with care! No out of range checks
    RMAGINE_INLINE_FUNCTION
    unsigned int operator[](const size_t& idx) const;

    RMAGINE_INLINE_FUNCTION
    unsigned int& operator[](const size_t& idx);
};

} // namespace rmagine


// Polyscope field access
unsigned int adaptorF_custom_accessVector3Value(
  const rmagine::Face& f, 
  unsigned int ind);

size_t adaptorF_size(const rmagine::Face& f);

namespace std
{

// Custom get function for MyStruct, index 0 (for `int`)
template <std::size_t I>
decltype(auto) get(rmagine::Face& f) {
    if constexpr (I == 0) return (f.v0);
    else if constexpr (I == 1) return (f.v1);
    else if constexpr (I == 2) return (f.v2);
    else static_assert(I < 3, "Index out of bounds");
}

template <std::size_t I>
decltype(auto) get(const rmagine::Face& f) {
    if constexpr (I == 0) return (f.v0);
    else if constexpr (I == 1) return (f.v1);
    else if constexpr (I == 2) return (f.v2);
    else static_assert(I < 3, "Index out of bounds");
}


} // namespace std




// namespace std
// {


// } // namespace std

#include "mesh_types.tcc"


#endif // RMAGINE_TYPES_MESH_TYPES_H