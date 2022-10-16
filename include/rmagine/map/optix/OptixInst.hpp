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
 * @brief OptixInst
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_MAP_OPTIX_INSTANCE_HPP
#define RMAGINE_MAP_OPTIX_INSTANCE_HPP


#include "optix_definitions.h"

#include "OptixGeometry.hpp"
#include "OptixScene.hpp"


struct OptixInstance;

namespace rmagine
{

class OptixInst
: public OptixGeometry
{
public:
    using Base = OptixGeometry;

    OptixInst(OptixContextPtr context = optix_default_context());

    virtual ~OptixInst();

    void set(OptixScenePtr geom);
    OptixScenePtr scene() const;

    virtual void apply();
    virtual void commit();
    virtual unsigned int depth() const;

    void setId(unsigned int id);
    unsigned int id() const;

    void disable();
    void enable();

    virtual OptixGeometryType type() const
    {
        return OptixGeometryType::INSTANCE;
    }

    const OptixInstance* data() const;

    OptixInstanceSBT sbt_data;
protected:
    // TODO hide this from header
    OptixInstance* m_data;

    // filled after commit
    // CUdeviceptr m_data_gpu = 0;
    // CUdeviceptr m_data_gpu;

    OptixScenePtr m_scene;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_INSTANCE_HPP