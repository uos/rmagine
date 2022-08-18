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

OptixAccelerationStructurePtr build_gas()
{
    OptixAccelerationStructurePtr as = std::make_shared<OptixAccelerationStructure>();
    return as;
}

void quickLaunch(
    CudaStreamPtr stream,
    const OptixAccelerationStructure& as,
    const OptixPipeline& pipeline,
    const OptixShaderBindingTable& sbt,
    Vector pos)
{
    
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
    Tbm[0].t = pos;
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

    std::cout << "- range: " << ranges[0] << std::endl;
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: Optix Multi Root" << std::endl;


    OptixContextPtr ctx = optix_default_context();
    CudaStreamPtr stream = ctx->getCudaContext()->createStream();

    std::vector<Vector> vertices;
    std::vector<Face> faces;

    genCube(vertices, faces);

    std::cout << "Cube " << vertices.size() << ", " << faces.size() << std::endl;

    Memory<Vector, RAM> vertices_ram1(vertices.size());
    std::copy(vertices.begin(), vertices.end(), vertices_ram1.raw());

    Memory<Vector, RAM> vertices_ram2(vertices_ram1.size());
    {
        Vector shift = {0.0, 5.0, 0.0};
        for(size_t i=0; i<vertices_ram2.size(); i++)
        {
            vertices_ram2[i] = vertices_ram1[i] + shift;
        }
    }

    Memory<Vector, RAM> vertices_ram3(vertices.size());
    {
        Vector shift = {0.0, 10.0, 0.0};
        for(size_t i=0; i<vertices_ram2.size(); i++)
        {
            vertices_ram3[i] = vertices_ram1[i] + shift;
        }
    }

    Memory<Vector, VRAM_CUDA> vertices_vram1 = vertices_ram1;
    Memory<Vector, VRAM_CUDA> vertices_vram2 = vertices_ram2;
    Memory<Vector, VRAM_CUDA> vertices_vram3 = vertices_ram3;

    CUdeviceptr tmp1 = reinterpret_cast<CUdeviceptr>(vertices_vram1.raw());
    CUdeviceptr tmp2 = reinterpret_cast<CUdeviceptr>(vertices_vram2.raw());
    CUdeviceptr tmp3 = reinterpret_cast<CUdeviceptr>(vertices_vram3.raw());


    Memory<Face, RAM> faces_ram(faces.size());
    std::copy(faces.begin(), faces.end(), faces_ram.raw());
    Memory<Face, VRAM_CUDA> faces_vram = faces_ram;


    // MULTI GEOMETRY:
    // Only care for:
    // - [ 2][       ERROR]: "buildInputs[2].type" != "buildInputs[0].type". 
    // - All build inputs for geometry acceleration structures must have the same type


    // 1. Compute ACC

    size_t gas1_n_elements = 3;
    OptixAccelerationStructure gas1;
    {
        // std::vector<OptixBuildInput> build_inputs;
        OptixBuildInput build_inputs[gas1_n_elements];

        { // 1
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            
            // VERTICES
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
            triangle_input.triangleArray.numVertices   = vertices_vram1.size();
            triangle_input.triangleArray.vertexBuffers = &tmp1;

            // FACES
            // std::cout << "- define faces" << std::endl;
            triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
            triangle_input.triangleArray.numIndexTriplets    = faces_vram.size();
            triangle_input.triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(faces_vram.raw());

            // ADDITIONAL SETTINGS
            triangle_input.triangleArray.flags         = (const uint32_t [1]) { 
                OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
            };
            // TODO: this is bad. I define the sbt records inside the sensor programs. 
            triangle_input.triangleArray.numSbtRecords = 1;

            build_inputs[0] = triangle_input;
        }

        { // 2
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            // VERTICES
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
            triangle_input.triangleArray.numVertices   = vertices_vram2.size();
            triangle_input.triangleArray.vertexBuffers = &tmp2;

            // FACES
            // std::cout << "- define faces" << std::endl;
            triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
            triangle_input.triangleArray.numIndexTriplets    = faces_vram.size();
            triangle_input.triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(faces_vram.raw());

            // ADDITIONAL SETTINGS
            triangle_input.triangleArray.flags         = (const uint32_t [1]) { 
                OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
            };

            // TODO: this is bad. I define the sbt records inside the sensor programs. 
            triangle_input.triangleArray.numSbtRecords = 1;
            // triangle_input.triangleArray.sbtIndexOffsetBuffer = 1;
            build_inputs[1] = triangle_input;
        }

        { // 3
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            // VERTICES
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
            triangle_input.triangleArray.numVertices   = vertices_vram3.size();
            triangle_input.triangleArray.vertexBuffers = &tmp3;

            // FACES
            // std::cout << "- define faces" << std::endl;
            triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
            triangle_input.triangleArray.numIndexTriplets    = faces_vram.size();
            triangle_input.triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(faces_vram.raw());

            // ADDITIONAL SETTINGS
            triangle_input.triangleArray.flags         = (const uint32_t [1]) { 
                OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
            };

            // TODO: this is bad. I define the sbt records inside the sensor programs. 
            triangle_input.triangleArray.numSbtRecords = 1;
            // triangle_input.triangleArray.sbtIndexOffsetBuffer = 1;
            build_inputs[2] = triangle_input;
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

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage(
                    ctx->ref(),
                    &accel_options,
                    build_inputs,
                    gas1_n_elements, // Number of build inputs
                    &gas_buffer_sizes
                    ) );

        
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_temp_buffer_gas ),
            gas_buffer_sizes.tempSizeInBytes) );
        
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &gas1.buffer ),
                    gas_buffer_sizes.outputSizeInBytes
                    ) );
        gas1.buffer_size = gas_buffer_sizes.outputSizeInBytes;

        OPTIX_CHECK( optixAccelBuild(
                    ctx->ref(),
                    stream->handle(),                  // CUDA stream
                    &accel_options,
                    build_inputs,
                    gas1_n_elements,                  // num build inputs
                    d_temp_buffer_gas,
                    gas_buffer_sizes.tempSizeInBytes,
                    gas1.buffer,
                    gas_buffer_sizes.outputSizeInBytes,
                    &gas1.handle,
                    nullptr,            // emitted property list
                    0                   // num emitted properties
                    ) );

        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
    }

    size_t gas2_n_elements = 2;
    OptixAccelerationStructure gas2;
    {
        // std::vector<OptixBuildInput> build_inputs;
        OptixBuildInput build_inputs[gas2_n_elements];

        { // 1
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            
            // VERTICES
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
            triangle_input.triangleArray.numVertices   = vertices_vram3.size();
            triangle_input.triangleArray.vertexBuffers = &tmp3;

            // FACES
            // std::cout << "- define faces" << std::endl;
            triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
            triangle_input.triangleArray.numIndexTriplets    = faces_vram.size();
            triangle_input.triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(faces_vram.raw());

            // ADDITIONAL SETTINGS
            triangle_input.triangleArray.flags         = (const uint32_t [1]) { 
                OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
            };
            // TODO: this is bad. I define the sbt records inside the sensor programs. 
            triangle_input.triangleArray.numSbtRecords = 1;

            build_inputs[0] = triangle_input;
        }

        { // 2
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            // VERTICES
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
            triangle_input.triangleArray.numVertices   = vertices_vram1.size();
            triangle_input.triangleArray.vertexBuffers = &tmp1;

            // FACES
            // std::cout << "- define faces" << std::endl;
            triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
            triangle_input.triangleArray.numIndexTriplets    = faces_vram.size();
            triangle_input.triangleArray.indexBuffer         = reinterpret_cast<CUdeviceptr>(faces_vram.raw());

            // ADDITIONAL SETTINGS
            triangle_input.triangleArray.flags         = (const uint32_t [1]) { 
                OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
            };

            // TODO: this is bad. I define the sbt records inside the sensor programs. 
            triangle_input.triangleArray.numSbtRecords = 1;
            // triangle_input.triangleArray.sbtIndexOffsetBuffer = 1;
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

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage(
                    ctx->ref(),
                    &accel_options,
                    build_inputs,
                    gas2_n_elements, // Number of build inputs
                    &gas_buffer_sizes
                    ) );

        
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_temp_buffer_gas ),
            gas_buffer_sizes.tempSizeInBytes) );
        
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &gas2.buffer ),
                    gas_buffer_sizes.outputSizeInBytes
                    ) );
        gas2.buffer_size = gas_buffer_sizes.outputSizeInBytes;

        OPTIX_CHECK( optixAccelBuild(
                    ctx->ref(),
                    stream->handle(),                  // CUDA stream
                    &accel_options,
                    build_inputs,
                    gas2_n_elements,                  // num build inputs
                    d_temp_buffer_gas,
                    gas_buffer_sizes.tempSizeInBytes,
                    gas2.buffer,
                    gas_buffer_sizes.outputSizeInBytes,
                    &gas2.handle,
                    nullptr,            // emitted property list
                    0                   // num emitted properties
                    ) );

        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
    }


    // [ 2][       ERROR]: "numBuildInputs" must be 1 for instance acceleration builds
    // size_t ias1_n_elements = 1;
    OptixAccelerationStructure ias1;
    {
        Memory<OptixInstance, RAM> inst_h(4);
        for(size_t i=0; i<inst_h.size(); i++)
        {
            OptixInstance& m_data = inst_h[i];

            m_data.instanceId = i;
            if(i % 2)
            {
                m_data.traversableHandle = gas1.handle;
            } else {
                m_data.traversableHandle = gas2.handle;
            }
            

            Transform T = Transform::Identity();
            T.t = {0.0, 0.0, static_cast<float>(i) * 2.0f};
            Matrix4x4 M;
            M.set(T);

            m_data.transform[ 0] = M(0,0); // Rxx
            m_data.transform[ 1] = M(0,1); // Rxy
            m_data.transform[ 2] = M(0,2); // Rxz
            m_data.transform[ 3] = M(0,3); // tx
            m_data.transform[ 4] = M(1,0); // Ryx
            m_data.transform[ 5] = M(1,1); // Ryy
            m_data.transform[ 6] = M(1,2); // Ryz
            m_data.transform[ 7] = M(1,3); // ty 
            m_data.transform[ 8] = M(2,0); // Rzx
            m_data.transform[ 9] = M(2,1); // Rzy
            m_data.transform[10] = M(2,2); // Rzz
            m_data.transform[11] = M(2,3); // tz

            m_data.sbtOffset = 0;
            m_data.visibilityMask = 255;
            m_data.flags = OPTIX_INSTANCE_FLAG_NONE;
        }

        CUdeviceptr m_inst_buffer;
        CUDA_CHECK( cudaMalloc( 
            reinterpret_cast<void**>( &m_inst_buffer ), 
            inst_h.size() * sizeof(OptixInstance) ) );

        CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( m_inst_buffer ),
                inst_h.raw(),
                inst_h.size() * sizeof(OptixInstance),
                cudaMemcpyHostToDevice,
                stream->handle()
                ) );

        // BEGIN WITH BUILD INPUT

    
        OptixBuildInput instance_input = {};
        instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.numInstances = inst_h.size();
        instance_input.instanceArray.instances = m_inst_buffer;

        OptixAccelBuildOptions ias_accel_options = {};
        unsigned int build_flags = OPTIX_BUILD_FLAG_NONE;
        { // BUILD FLAGS
            build_flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            build_flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
            #if OPTIX_VERSION >= 73000
            build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
            #endif
        }

        ias_accel_options.buildFlags = build_flags;
        ias_accel_options.motionOptions.numKeys = 1;
        ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;


        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( 
            ctx->ref(), 
            &ias_accel_options,
            &instance_input, 
            1, 
            &ias_buffer_sizes ) );

        CUdeviceptr d_temp_buffer_ias;
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_temp_buffer_ias ),
            ias_buffer_sizes.tempSizeInBytes) );

        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &ias1.buffer ),
            ias_buffer_sizes.outputSizeInBytes
        ));

        ias1.buffer_size = ias_buffer_sizes.outputSizeInBytes;

        OPTIX_CHECK(optixAccelBuild( 
            ctx->ref(), 
            stream->handle(), 
            &ias_accel_options, 
            &instance_input, 
            1, // num build inputs
            d_temp_buffer_ias,
            ias_buffer_sizes.tempSizeInBytes, 
            ias1.buffer,
            ias_buffer_sizes.outputSizeInBytes,
            &ias1.handle,
            nullptr, 
            0 
        ));
    }



    
    std::vector<CUdeviceptr> instances;

    CUdeviceptr instance_ptrs;
    {
        size_t N = 4;
        for(size_t i=0; i<N; i++)
        {
            OptixInstance m_data;

            m_data.instanceId = i;
            if(i % 2)
            {
                m_data.traversableHandle = gas1.handle;
            } else {
                m_data.traversableHandle = gas2.handle;
            }
            

            Transform T = Transform::Identity();
            T.t = {0.0, 0.0, static_cast<float>(i) * 2.0f};
            Matrix4x4 M;
            M.set(T);

            m_data.transform[ 0] = M(0,0); // Rxx
            m_data.transform[ 1] = M(0,1); // Rxy
            m_data.transform[ 2] = M(0,2); // Rxz
            m_data.transform[ 3] = M(0,3); // tx
            m_data.transform[ 4] = M(1,0); // Ryx
            m_data.transform[ 5] = M(1,1); // Ryy
            m_data.transform[ 6] = M(1,2); // Ryz
            m_data.transform[ 7] = M(1,3); // ty 
            m_data.transform[ 8] = M(2,0); // Rzx
            m_data.transform[ 9] = M(2,1); // Rzy
            m_data.transform[10] = M(2,2); // Rzz
            m_data.transform[11] = M(2,3); // tz

            m_data.sbtOffset = 0;
            m_data.visibilityMask = 255;
            m_data.flags = OPTIX_INSTANCE_FLAG_NONE;

            CUdeviceptr inst_ptr;
            CUDA_CHECK( cudaMalloc( 
                reinterpret_cast<void**>( &inst_ptr ), 
                sizeof(OptixInstance) ) );

            CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( inst_ptr ),
                &m_data,
                sizeof(OptixInstance),
                cudaMemcpyHostToDevice,
                stream->handle()
                ) );
            instances.push_back(inst_ptr);
        }

        CUDA_CHECK( cudaMalloc( 
                reinterpret_cast<void**>( &instance_ptrs ), 
                sizeof(OptixInstance*) * N ) );

        CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( instance_ptrs ),
                &instances[0],
                sizeof(OptixInstance*) * N,
                cudaMemcpyHostToDevice,
                stream->handle()
                ) );

        
    }

    OptixAccelerationStructure ias2;
    {
        // BEGIN WITH BUILD INPUT
        OptixBuildInput instance_input = {};
        instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS;
        instance_input.instanceArray.numInstances = instances.size();
        instance_input.instanceArray.instances = instance_ptrs;

        OptixAccelBuildOptions ias_accel_options = {};
        unsigned int build_flags = OPTIX_BUILD_FLAG_NONE;
        { // BUILD FLAGS
            build_flags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            build_flags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
            #if OPTIX_VERSION >= 73000
            build_flags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS;
            #endif
        }

        ias_accel_options.buildFlags = build_flags;
        ias_accel_options.motionOptions.numKeys = 1;
        ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;


        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( 
            ctx->ref(), 
            &ias_accel_options,
            &instance_input, 
            1, 
            &ias_buffer_sizes ) );

        CUdeviceptr d_temp_buffer_ias;
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_temp_buffer_ias ),
            ias_buffer_sizes.tempSizeInBytes) );

        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &ias2.buffer ),
            ias_buffer_sizes.outputSizeInBytes
        ));

        ias2.buffer_size = ias_buffer_sizes.outputSizeInBytes;

        OPTIX_CHECK(optixAccelBuild( 
            ctx->ref(), 
            stream->handle(), 
            &ias_accel_options, 
            &instance_input, 
            1, // num build inputs
            d_temp_buffer_ias,
            ias_buffer_sizes.tempSizeInBytes, 
            ias2.buffer,
            ias_buffer_sizes.outputSizeInBytes,
            &ias2.handle,
            nullptr, 
            0 
        ));
    }



    // Build pipeline
    OptixPipeline pipeline = nullptr;

    // required for SBT
    OptixProgramGroup hitgroup_prog_group = nullptr;
    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    {
        // Create pipeline
        // independent from 

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
    }


    OptixShaderBindingTable sbt = {};
    {

        // 4. setup shader binding table
        typedef SbtRecord<RayGenDataEmpty>     RayGenSbtRecord;
        typedef SbtRecord<MissDataEmpty>       MissSbtRecord;
        typedef SbtRecord<HitGroupDataScene>   HitGroupSbtRecord;        

        sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
        sbt.missRecordCount             = 1;
        // sbt.missRecordCount             = 1000; // works as well
        sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
        sbt.hitgroupRecordCount         = gas1_n_elements;
        // sbt.hitgroupRecordCount         = 1000; // works as well


        const size_t raygen_record_size     = sizeof( RayGenSbtRecord );
        const size_t miss_record_size       = sizeof( MissSbtRecord ) * sbt.missRecordCount;
        const size_t hitgroup_record_size   = sizeof( HitGroupSbtRecord ) * sbt.hitgroupRecordCount;


        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.raygenRecord ), raygen_record_size) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.missRecordBase ), miss_record_size ) );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.hitgroupRecordBase ), hitgroup_record_size ) );


        {   
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( sbt.raygenRecord ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );
        }
        
        {
            MissSbtRecord ms_sbt[sbt.missRecordCount];
            for(size_t i=0; i<sbt.hitgroupRecordCount; i++)
            {
                OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt[i] ) );
            }
            CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( sbt.missRecordBase ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
                ) );
        }
        
        {
            // per primitive geometry?? yes tested
            // - knowledge of map required?
            // - and also of program
            // maybe thats why it called "binding" table
            HitGroupSbtRecord hg_sbt[sbt.hitgroupRecordCount];
            for(size_t i=0; i<sbt.hitgroupRecordCount; i++)
            {
                OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt[i] ) );
            }
            
            CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( sbt.hitgroupRecordBase ),
                &hg_sbt,
                hitgroup_record_size,
                cudaMemcpyHostToDevice
                ));
        }
    }

    



    // run

    std::cout << "------ GAS1 - LAUNCH 1/3 -------" << std::endl;
    quickLaunch(stream, gas1, pipeline, sbt, {-5.0, 0.0, 0.0});

    std::cout << "------ GAS1 - LAUNCH 2/3 -------" << std::endl;
    quickLaunch(stream, gas1, pipeline, sbt, {-5.0, 5.0, 0.0});

    std::cout << "------ GAS1 - LAUNCH 3/3 -------" << std::endl;
    quickLaunch(stream, gas1, pipeline, sbt, {-5.0, 10.0, 0.0});


    std::cout << "------ GAS2 - LAUNCH 1/2 -------" << std::endl;
    quickLaunch(stream, gas2, pipeline, sbt, {-5.0, 0.0, 0.0});

    std::cout << "------ GAS2 - LAUNCH 2/2 -------" << std::endl;
    quickLaunch(stream, gas2, pipeline, sbt, {-5.0, 10.0, 0.0});



    for(size_t i=0; i<4; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            std::cout << "------ IAS1 - LAUNCH (" << i << "," << j << ") -> (3,3) -------" << std::endl;
            quickLaunch(stream, ias1, pipeline, sbt, {
                -5.0, 
                5.0f * static_cast<float>(j), 
                2.0f * static_cast<float>(i)
            });
        }
    }




    for(size_t i=0; i<4; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            std::cout << "------ IAS2 - LAUNCH (" << i << "," << j << ") -> (3,3) -------" << std::endl;
            quickLaunch(stream, ias2, pipeline, sbt, {
                -5.0, 
                5.0f * static_cast<float>(j), 
                2.0f * static_cast<float>(i)
            });
        }
    }


    return 0;
}