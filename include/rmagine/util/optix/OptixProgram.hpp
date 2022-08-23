/**
 * Copyright (c) 2021, University Osnabrück
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

/*
 * OptixProgram.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: Alexander Mock
 */

#ifndef RMAGINE_OPTIX_UTIL_OPTIX_PROGRAM_HPP
#define RMAGINE_OPTIX_UTIL_OPTIX_PROGRAM_HPP

#include <optix.h>
#include <memory>

#include "OptixSbtRecord.hpp"
#include "OptixData.hpp"
#include <rmagine/map/optix/optix_sbt.h>


namespace rmagine {

struct RayGenModule
{
    using RayGenData        = RayGenDataEmpty;
    using RayGenSbtRecord   = SbtRecord<RayGenData>;

    OptixModule       module       = nullptr;
    OptixProgramGroup prog_group   = nullptr;

    // TODO SBT RECORD
    RayGenSbtRecord*    record_h      = nullptr;
    CUdeviceptr 	    record        = 0;
    

    ~RayGenModule();
};

using RayGenModulePtr = std::shared_ptr<RayGenModule>;

struct HitModule
{
    using MissData          = MissDataEmpty;
    using HitGroupData      = OptixSceneSBT;
    using MissSbtRecord     = SbtRecord<MissData>;
    using HitGroupSbtRecord = SbtRecord<HitGroupData>;

    OptixModule         module              = nullptr;
    OptixProgramGroup   prog_group_hit      = nullptr;
    OptixProgramGroup   prog_group_miss     = nullptr;

    MissSbtRecord*      record_miss_h       = nullptr;
    CUdeviceptr 	    record_miss         = 0;
    unsigned int 	    record_miss_stride  = 0;
    unsigned int 	    record_miss_count   = 0;

    HitGroupSbtRecord*  record_hit_h        = nullptr;
    CUdeviceptr 	    record_hit          = 0;
    unsigned int 	    record_hit_stride   = 0;
    unsigned int 	    record_hit_count    = 0;

    ~HitModule();
};

using HitModulePtr = std::shared_ptr<HitModule>;

struct OptixSBT 
{
    OptixShaderBindingTable sbt = {};
    
    ~OptixSBT();
};

using OptixSBTPtr = std::shared_ptr<OptixSBT>;

struct OptixSensorPipeline
{
    OptixPipeline pipeline = nullptr;

    ~OptixSensorPipeline();
};

using OptixSensorPipelinePtr = std::shared_ptr<OptixSensorPipeline>;


class OptixProgram
{
public:
    virtual ~OptixProgram();
    
    virtual void updateSBT() {};

    OptixModule module = nullptr;
    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
};

using OptixProgramPtr = std::shared_ptr<OptixProgram>;

} // namespace rmagine

#endif // RMAGINE_OPTIX_UTIL_OPTIX_PROGRAM_HPP