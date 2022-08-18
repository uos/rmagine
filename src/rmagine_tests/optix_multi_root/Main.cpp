#include <iostream>

#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/util/StopWatch.hpp>
#include <fstream>

#include <rmagine/util/synthetic.h>

#include <rmagine/types/MemoryCuda.hpp>

#include <rmagine/util/optix/OptixContext.hpp>

#include <optix_stubs.h>

#include "rmagine/util/optix/OptixUtil.hpp"
#include "rmagine/util/optix/OptixSbtRecord.hpp"
#include "rmagine/util/optix/OptixData.hpp"

using namespace rmagine;


Memory<LiDARModel, RAM> velodyne_model()
{
    Memory<LiDARModel, RAM> model(1);
    model->theta.min = -M_PI;
    model->theta.inc = 0.4 * M_PI / 180.0;
    model->theta.size = 900;

    model->phi.min = -15.0 * M_PI / 180.0;
    model->phi.inc = 2.0 * M_PI / 180.0;
    model->phi.size = 16;
    
    model->range.min = 0.5;
    model->range.max = 130.0;
    return model;
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: Optix Multi Root" << std::endl;

    std::vector<Vector> vertices;
    std::vector<Face> faces;

    genCube(vertices, faces);

    std::cout << "Cube " << vertices.size() << ", " << faces.size() << std::endl;

    Memory<Vector, RAM> vertices_ram(vertices.size());
    std::copy(vertices.begin(), vertices.end(), vertices_ram.raw());

    Memory<Vector, RAM> vertices_ram2(vertices.size());
    Vector shift = {0.0, 5.0, 0.0};
    for(size_t i=0; i<vertices_ram2.size(); i++)
    {
        vertices_ram2[i] = vertices_ram[i] + shift;
    }


    Memory<Face, RAM> faces_ram(faces.size());
    std::copy(faces.begin(), faces.end(), faces_ram.raw());


    Memory<Vector, VRAM_CUDA> vertices_vram = vertices_ram;
    Memory<Vector, VRAM_CUDA> vertices_vram2 = vertices_ram2;

    CUdeviceptr tmp1 = reinterpret_cast<CUdeviceptr>(vertices_vram.raw());
    CUdeviceptr tmp2 = reinterpret_cast<CUdeviceptr>(vertices_vram2.raw());

    Memory<Face, VRAM_CUDA> faces_vram = faces_ram;


    OptixContextPtr ctx = optix_default_context();

    CudaStreamPtr stream = ctx->getCudaContext()->createStream();


    OptixBuildInput build_inputs[2];

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

    { // 1
        OptixBuildInput triangle_input = {};
        triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        

        // VERTICES
        triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
        triangle_input.triangleArray.numVertices   = vertices_vram.size();
        triangle_input.triangleArray.vertexBuffers = &tmp1;

        // FACES
        // std::cout << "- define faces" << std::endl;
        triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
        triangle_input.triangleArray.numIndexTriplets    = faces_vram.size();
        triangle_input.triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(faces_vram.raw());

        // ADDITIONAL SETTINGS
        triangle_input.triangleArray.flags         = triangle_input_flags;
        // TODO: this is bad. I define the sbt records inside the sensor programs. 
        triangle_input.triangleArray.numSbtRecords = 1;

        build_inputs[0] = triangle_input;
    }

    {
        OptixBuildInput triangle_input = {};
        triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        

        // VERTICES
        triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
        triangle_input.triangleArray.numVertices   = vertices_vram.size();
        triangle_input.triangleArray.vertexBuffers = &tmp2;

        // FACES
        // std::cout << "- define faces" << std::endl;
        triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
        triangle_input.triangleArray.numIndexTriplets    = faces_vram.size();
        triangle_input.triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(faces_vram.raw());

        // ADDITIONAL SETTINGS
        triangle_input.triangleArray.flags         = triangle_input_flags;
        // TODO: this is bad. I define the sbt records inside the sensor programs. 
        triangle_input.triangleArray.numSbtRecords = 1;
        triangle_input.triangleArray.sbtIndexOffsetBuffer = 1;
        triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(int);
        triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(int);

        build_inputs[1] = triangle_input;
    }

    // Acceleration Options
    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};

    unsigned int build_flags = OPTIX_BUILD_FLAG_NONE;

    { // BUILD FLAGS
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    }

    accel_options.buildFlags = build_flags;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    

    OptixAccelerationStructure as;
    

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                ctx->ref(),
                &accel_options,
                build_inputs,
                2, // Number of build inputs
                &gas_buffer_sizes
                ) );

    
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes) );
    
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &as.buffer ),
                gas_buffer_sizes.outputSizeInBytes
                ) );
    as.buffer_size = gas_buffer_sizes.outputSizeInBytes;

    OPTIX_CHECK( optixAccelBuild(
                ctx->ref(),
                stream->handle(),                  // CUDA stream
                &accel_options,
                build_inputs,
                2,                  // num build inputs
                d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                as.buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &as.handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );


    // SCENE FINISHED


    const char *kernel =
    #include "kernels/SphereProgramRangesString.h"
    ;

     // 1. INIT MODULE
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof( log );

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
    
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur        = false;

    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeline_compile_options.numPayloadValues      = 1;
    pipeline_compile_options.numAttributeValues    = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

    pipeline_compile_options.pipelineLaunchParamsVariableName = "mem";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    std::string ptx(kernel);

    if(ptx.empty())
    {
        throw std::runtime_error("ScanProgramRanges could not find its PTX part");
    }

    OptixModule module = nullptr;
    

    OPTIX_CHECK( optixModuleCreateFromPTX(
                ctx->ref(),
                &module_compile_options,
                &pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
                ));




    // 2. initProgramGroups
    OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    sizeof_log = sizeof( log );

    OptixProgramGroup raygen_prog_group   = nullptr;
    optixProgramGroupCreate(
                ctx->ref(),
                &raygen_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &raygen_prog_group
                );

    

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof( log );
    
    OptixProgramGroup miss_prog_group     = nullptr;
    optixProgramGroupCreate(
                ctx->ref(),
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &miss_prog_group
                );


    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    sizeof_log = sizeof( log );
    OptixProgramGroup hitgroup_prog_group = nullptr;
    optixProgramGroupCreate(
                ctx->ref(),
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &hitgroup_prog_group
                );

    OptixProgramGroup program_groups[] = { 
        raygen_prog_group, 
        miss_prog_group, 
        hitgroup_prog_group 
    };



    // 3. link pipeline
    // traverse depth = 2 for ias + gas
    uint32_t    max_traversable_depth = 3;
    const uint32_t    max_trace_depth  = 1;


    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = max_trace_depth;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    sizeof_log = sizeof( log );

    OptixPipeline pipeline = nullptr;
    OPTIX_CHECK_LOG( optixPipelineCreate(
                ctx->ref(),
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                log,
                &sizeof_log,
                &pipeline
                ) );


    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                0,  // maxCCDepth
                                                0,  // maxDCDEpth
                                                &direct_callable_stack_size_from_traversal,
                                                &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            max_traversable_depth  // maxTraversableDepth
                                            ) );


    // 4. setup shader binding table
    typedef SbtRecord<RayGenDataEmpty>     RayGenSbtRecord;
    typedef SbtRecord<MissDataEmpty>       MissSbtRecord;
    typedef SbtRecord<HitGroupDataEmpty>   HitGroupSbtRecord;

    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof( RayGenSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof( MissSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
    MissSbtRecord ms_sbt;
    
    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( miss_record ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
                ) );

    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( hitgroup_record ),
                &hg_sbt,
                hitgroup_record_size,
                cudaMemcpyHostToDevice
                ) );

    
    OptixShaderBindingTable sbt = {};

    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = miss_record;
    sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    sbt.hitgroupRecordCount         = 1;


    // run

    Memory<Transform, RAM> Tsb(1);
    Tsb[0] = Transform::Identity();
    Memory<Transform, VRAM_CUDA> Tsb_ = Tsb;
    
    Memory<SphericalModel, RAM> model(1);
    model[0].phi.size = 1;
    model[0].phi.min = 0.0;
    model[0].theta.size = 1;
    model[0].theta.min = 0.0;
    model[0].range.min = 0.0;
    model[0].range.max = 100.0;
    Memory<SphericalModel, VRAM_CUDA> model_ = model;

    Memory<Transform, RAM> Tbm(1);
    Tbm[0] = Transform::Identity();
    Tbm[0].t.x = -5.0;
    Tbm[0].t.y = 5.0;
    Memory<Transform, VRAM_CUDA> Tbm_ = Tbm;

    Memory<float, VRAM_CUDA> ranges_(1);




    Memory<OptixSimulationDataRangesSphere, RAM> mem(1);
    mem->Tsb = Tsb_.raw();
    mem->model = model_.raw();
    mem->Tbm = Tbm_.raw();
    mem->ranges = ranges_.raw();
    mem->handle = as.handle;

    Memory<OptixSimulationDataRangesSphere, VRAM_CUDA> d_mem(1);
    copy(mem, d_mem, stream->handle());



    std::cout << "LAUNCH!" << std::endl;

    OPTIX_CHECK( optixLaunch(
                pipeline,
                stream->handle(),
                reinterpret_cast<CUdeviceptr>(d_mem.raw()), 
                sizeof( OptixSimulationDataRangesSphere ),
                &sbt,
                1, // width Xdim
                1, // height Ydim
                1 // depth Zdim
                ));

    // download
    Memory<float, RAM> ranges = ranges_;

    std::cout << "Range: " << ranges[0] << std::endl;


    return 0;
}