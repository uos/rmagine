/*
 * Copyright (c) 2021, University Osnabrück. 
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
 * @brief Definition of attributes that are computed during generic simulation
 *
 * @date 03.01.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2021, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 */

#ifndef RMAGINE_SIMULATION_RESULTS_HPP
#define RMAGINE_SIMULATION_RESULTS_HPP

#include <rmagine/types/Bundle.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/math/types.h>

namespace rmagine
{

/**
 * @brief Whether a hit occured or not
 * 
 * @tparam MemT Use MemT=RAM for Embree and MemT=VRAM_CUDA for Optix
 */
template<typename MemT>
struct Hits {
    Memory<uint8_t, MemT> hits;
};

/**
 * @brief Ranges computed by the simulators
 * 
 * @tparam MemT Use MemT=RAM for Embree and MemT=VRAM_CUDA for Optix
 */
template<typename MemT>
struct Ranges {
    Memory<float, MemT> ranges;
};

/**
 * @brief Points (x,y,z) computed by the simulators
 * 
 * @tparam MemT Use MemT=RAM for Embree and MemT=VRAM_CUDA for Optix
 */
template<typename MemT>
struct Points {
    Memory<Point, MemT> points;
};

/**
 * @brief Normals (x,y,z) computed by the simulators
 * 
 * @tparam MemT 
 */
template<typename MemT>
struct Normals {
    Memory<Vector, MemT> normals;
};

/**
 * @brief FaceIds computed by the simulators
 * 
 * @tparam MemT 
 */
template<typename MemT>
struct FaceIds {
    Memory<unsigned int, MemT> face_ids;
};


/**
 * @brief GeomIds computed by the simulators
 * 
 * Embree:
 * - each instance can have multiply geometries 
 * -> id of those
 * 
 * OptiX:
 * - each instance can have only one geometry
 * -> id is zero anytime
 * 
 * @tparam MemT 
 */
template<typename MemT>
struct GeomIds {
    Memory<unsigned int, MemT> geom_ids;
};

/**
 * @brief ObjectIds computed by the simulators
 * 
 * @tparam MemT 
 */
template<typename MemT>
struct ObjectIds {
    Memory<unsigned int, MemT> object_ids;
};


template<typename MemT>
using IntAttrAny = Bundle<
    Hits<MemT>,
    Ranges<MemT>,
    Points<MemT>,
    Normals<MemT>,
    FaceIds<MemT>,
    ObjectIds<MemT>
>;

/**
 * @brief Helper function to resize a whole bundle of attributes by one size
 * 
 * @tparam MemT 
 * @tparam BundleT 
 * @param res 
 * @param W 
 * @param H 
 * @param N 
 */
template<typename MemT, typename BundleT>
void resizeMemoryBundle(BundleT& res, 
    unsigned int W,
    unsigned int H,
    unsigned int N )
{
    if constexpr(BundleT::template has<Hits<MemT> >())
    {
        res.Hits<MemT>::hits.resize(W*H*N);
    }

    if constexpr(BundleT::template has<Ranges<MemT> >())
    {
        res.Ranges<MemT>::ranges.resize(W*H*N);
    }

    if constexpr(BundleT::template has<Points<MemT> >())
    {
        res.Points<MemT>::points.resize(W*H*N);
    }

    if constexpr(BundleT::template has<Normals<MemT> >())
    {
        res.Normals<MemT>::normals.resize(W*H*N);
    }

    if constexpr(BundleT::template has<FaceIds<MemT> >())
    {
        res.FaceIds<MemT>::face_ids.resize(W*H*N);
    }

    if constexpr(BundleT::template has<ObjectIds<MemT> >())
    {
        res.ObjectIds<MemT>::object_ids.resize(W*H*N);
    }
}


} // namespace rmagine

#endif // RMAGINE_SIMULATION_RESULTS_HPP