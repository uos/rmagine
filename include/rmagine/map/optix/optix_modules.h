#ifndef RMAGINE_MAP_OPTIX_MODULES_H
#define RMAGINE_MAP_OPTIX_MODULES_H

#include <optix.h>
#include <memory>

#include <rmagine/util/optix/OptixSbtRecord.hpp>
#include <rmagine/util/optix/OptixData.hpp>

#include "optix_sbt.h"

namespace rmagine
{

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

struct OptixSensorProgram
{
    OptixSensorPipelinePtr pipeline;
    OptixSBTPtr            sbt;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_MODULES_H