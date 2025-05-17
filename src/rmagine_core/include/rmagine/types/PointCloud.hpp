/*
 * Copyright (c) 2024, University Osnabrück
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
 * @brief PointCloud classes
 *
 * @date 03.10.2024
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2024, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_TYPES_POINT_CLOUD_H
#define RMAGINE_TYPES_POINT_CLOUD_H

#include <rmagine/types/Memory.hpp>
#include <rmagine/math/types.h>

namespace rmagine
{

template<typename MemT>
struct PointCloud_
{
  Memory<Vector, MemT>        points;
  Memory<uint8_t, MemT>       mask;
  Memory<Vector, MemT>        normals;
  Memory<unsigned int, MemT>  ids;
  Memory<float, MemT>         time_offsets; // in seconds
};

using PointCloud = PointCloud_<RAM>;

template<typename MemT>
struct PointCloudView_
{
  MemoryView<Vector, MemT>        points; // required
  MemoryView<uint8_t, MemT>       mask         = MemoryView<uint8_t, MemT>::Empty();
  MemoryView<Vector, MemT>        normals      = MemoryView<Vector, MemT>::Empty();
  MemoryView<unsigned int, MemT>  ids          = MemoryView<uint32_t, MemT>::Empty();
  MemoryView<float, MemT>         time_offsets = MemoryView<float, MemT>::Empty();
};

// default: RAM
using PointCloudView = PointCloudView_<RAM>;

template<typename MemTto, typename MemTfrom>
PointCloud_<MemTto> transfer(const PointCloudView_<MemTfrom>& from)
{
  return PointCloud_<MemTto>{
    .points       = from.points,
    .mask         = from.mask,
    .normals      = from.normals,
    .ids          = from.ids,
    .time_offsets = from.time_offsets
  };
}

template<typename MemT>
const PointCloudView_<MemT> watch(const PointCloud_<MemT>& from)
{
  return {
    .points       = from.points,
    .mask         = from.mask,
    .normals      = from.normals,
    .ids          = from.ids,
    .time_offsets = from.time_offsets
  };
}

} // namespace rmagine


#endif // RMAGINE_TYPES_MESH_TYPES_H