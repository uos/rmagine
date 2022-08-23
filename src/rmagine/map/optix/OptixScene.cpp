#include "rmagine/map/optix/OptixScene.hpp"
#include "rmagine/map/optix/OptixGeometry.hpp"
#include "rmagine/map/optix/OptixMesh.hpp"
#include "rmagine/map/optix/OptixInst.hpp"
#include "rmagine/map/optix/OptixAccelerationStructure.hpp"

#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/util/optix/OptixDebug.hpp>

#include <optix_stubs.h>

#include <rmagine/math/assimp_conversions.h>
#include <rmagine/util/assimp/helper.h>
#include <rmagine/math/linalg.h>

// use own lib instead
#include "rmagine/util/optix/OptixUtil.hpp"
#include "rmagine/util/optix/OptixSbtRecord.hpp"
#include "rmagine/util/optix/OptixData.hpp"



namespace rmagine
{

static std::string hit_ptx()
{
    static const char* kernel = 
    #include "kernels/ProgramHitString.h"
    ;
    return std::string(kernel);
}

static std::string raygen_ptx_from_model_type(unsigned int model_type)
{
    std::string ptx;

    if(model_type == 0)
    {
        static const char* kernel =
        #include "kernels/SphereProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else if(model_type == 1) {
        const char *kernel =
        #include "kernels/PinholeProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else if(model_type == 2) {
        const char *kernel =
        #include "kernels/O1DnProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else if(model_type == 3) {
        const char *kernel =
        #include "kernels/OnDnProgramGenString.h"
        ;
        ptx = std::string(kernel);
    } else {
        std::cout << "[OptixScene::raygen_ptx_from_model_type] ERROR model_type " << model_type << " not supported!" << std::endl;
        throw std::runtime_error("[OptixScene::raygen_ptx_from_model_type] ERROR loading ptx");
    }

    return ptx;
}


static std::vector<OptixModuleCompileBoundValueEntry> make_bounds(
    const OptixSimulationDataGeneric& flags)
{
    std::vector<OptixModuleCompileBoundValueEntry> options;
        
    { // computeHits
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeHits);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeHits );
        option.boundValuePtr = &flags.computeHits;
        options.push_back(option);
    }
    
    { // computeRanges
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeRanges);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeRanges );
        option.boundValuePtr = &flags.computeRanges;
        options.push_back(option);
    }

    { // computePoints
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computePoints);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computePoints );
        option.boundValuePtr = &flags.computePoints;
        options.push_back(option);
    }

    { // computeNormals
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeNormals);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeNormals );
        option.boundValuePtr = &flags.computeNormals;
        options.push_back(option);
    }

    { // computeFaceIds
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeFaceIds);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeFaceIds );
        option.boundValuePtr = &flags.computeFaceIds;
        options.push_back(option);
    }

    { // computeGeomIds
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeGeomIds);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeGeomIds );
        option.boundValuePtr = &flags.computeGeomIds;
        options.push_back(option);
    }

    { // computeObjectIds
        OptixModuleCompileBoundValueEntry option = {};
        option.pipelineParamOffsetInBytes = offsetof(OptixSimulationDataGeneric, computeObjectIds);
        option.sizeInBytes = sizeof( OptixSimulationDataGeneric::computeObjectIds );
        option.boundValuePtr = &flags.computeObjectIds;
        options.push_back(option);
    }

    return options;
}




OptixScene::OptixScene(OptixContextPtr context)
:OptixEntity(context)
{
    m_pipeline_compile_options = {};
    m_pipeline_compile_options.usesMotionBlur        = false;


    m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    
    // max payload values: 32
    m_pipeline_compile_options.numPayloadValues      = 0;
    // if dont use module payloads:
    // pipeline_compile_options.numPayloadValues      = 8;
    m_pipeline_compile_options.numAttributeValues    = 2;
#ifndef NDEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "mem";
    m_pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    m_payload_type.numPayloadValues = 8;
    m_payload_type.payloadSemantics = m_semantics;



    m_program_group_options = {};
    m_program_group_options.payloadType = &m_payload_type;

}

OptixScene::~OptixScene()
{
    if(sbt_data.geometries)
    {
        cudaFree(sbt_data.geometries);
    }

    for(auto elem : m_geometries)
    {
        elem.second->cleanupParents();
    }
}

unsigned int OptixScene::add(OptixGeometryPtr geom)
{
    auto it = m_ids.find(geom);
    if(it == m_ids.end())
    {
        if(m_geometries.empty())
        {
            // first element
            m_geom_type = geom->type();
            if(m_geom_type == OptixGeometryType::INSTANCE)
            {
                m_type = OptixSceneType::INSTANCES;
            } else {
                m_type = OptixSceneType::GEOMETRIES;
            }
        }

        if(geom->type() != m_geom_type)
        {
            std::cout << "[OptixScene::add()] WARNING - used mixed types of geometries. NOT supported" << std::endl;
        }

        // add geom to self
        unsigned int id = gen.get();
        m_geometries[id] = geom;
        m_ids[geom] = id;

        // add self to geom
        geom->addParent(this_shared<OptixScene>());

        m_geom_added = true;

        return id;
    } else {
        return it->second;
    }
}

unsigned int OptixScene::get(OptixGeometryPtr geom) const
{
    return m_ids.at(geom);
}

std::optional<unsigned int> OptixScene::getOpt(OptixGeometryPtr geom) const
{
    auto it = m_ids.find(geom);
    if(it != m_ids.end())
    {
        return it->second;
    }

    return {};
}

bool OptixScene::has(OptixGeometryPtr geom) const
{
    return (m_ids.find(geom) != m_ids.end());
}

bool OptixScene::has(unsigned int geom_id) const
{
    return (m_geometries.find(geom_id) != m_geometries.end());
}

bool OptixScene::remove(OptixGeometryPtr geom)
{
    bool ret = false;

    auto it = m_ids.find(geom);
    if(it != m_ids.end())
    {
        unsigned int geom_id = it->second;

        m_ids.erase(it);
        m_geometries.erase(geom_id);
        geom->removeParent(this_shared<OptixScene>());
        gen.give_back(geom_id);

        m_geom_removed = true;
        ret = true;
    }

    return ret;
}

OptixGeometryPtr OptixScene::remove(unsigned int geom_id)
{
    OptixGeometryPtr ret;

    auto it = m_geometries.find(geom_id);
    if(it != m_geometries.end())
    {
        OptixGeometryPtr geom = it->second;

        m_geometries.erase(it);
        m_ids.erase(geom);
        geom->removeParent(this_shared<OptixScene>());
        gen.give_back(geom_id);

        m_geom_removed = true;

        ret = geom;
    }

    return ret;
}

std::map<unsigned int, OptixGeometryPtr> OptixScene::geometries() const
{
    return m_geometries;
}

std::unordered_map<OptixGeometryPtr, unsigned int> OptixScene::ids() const
{
    return m_ids;
}

void OptixScene::commit()
{
    if(m_type == OptixSceneType::INSTANCES)
    {
        buildIAS();
    } else if(m_type == OptixSceneType::GEOMETRIES ) {
        buildGAS();
    }

    if(m_pipeline_compile_options.traversableGraphFlags != m_traversable_graph_flags)
    {
        // TODO: need to update pipelines (nearly everything)

        if(m_pipelines.size() > 0)
        {
            std::cout << "[OptixScene::commit()] NEED TO UPDATE PIPELINES! Not implemented." << std::endl;
        }

        m_pipeline_compile_options.traversableGraphFlags = m_traversable_graph_flags;
    } else {
        updateSBT();
    }
}

OptixInstPtr OptixScene::instantiate()
{
    OptixInstPtr ret = std::make_shared<OptixInst>();
    ret->set(this_shared<OptixScene>());
    return ret;
}

void OptixScene::cleanupParents()
{
    for(auto it = m_parents.cbegin(); it != m_parents.cend();)
    {
        if (it->expired())
        {
            m_parents.erase(it++);    // or "it = m.erase(it)" since C++11
        } else {
            ++it;
        }
    }
}

std::unordered_set<OptixInstPtr> OptixScene::parents() const
{
    std::unordered_set<OptixInstPtr> ret;

    for(OptixInstWPtr elem : m_parents)
    {
        if(auto tmp = elem.lock())
        {
            ret.insert(tmp);
        }
    }
    
    return ret;
}

void OptixScene::addParent(OptixInstPtr parent)
{
    m_parents.insert(parent);
}

void OptixScene::buildGAS()
{
    // std::cout << "SCENE BUILD GAS" << std::endl;
    size_t n_build_inputs = m_geometries.size();

    OptixBuildInput build_inputs[n_build_inputs];

    OptixSceneSBT sbt_data_h;
    cudaMallocHost(&sbt_data_h.geometries, sizeof(OptixGeomSBT) * n_build_inputs);

    // TODO make proper realloc
    sbt_data.n_geometries = n_build_inputs;
    sbt_data.type = m_type;
    CUDA_CHECK( cudaMalloc(&sbt_data.geometries, sizeof(OptixGeomSBT) * n_build_inputs) );

    size_t idx = 0;
    for(auto elem : m_geometries)
    {
        OptixMeshPtr mesh = std::dynamic_pointer_cast<OptixMesh>(elem.second);
        if(mesh)
        {
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            // VERTICES
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes = sizeof(Point);
            triangle_input.triangleArray.numVertices   = mesh->vertices.size();
            triangle_input.triangleArray.vertexBuffers = mesh->getVertexBuffer();
        
            // FACES
            triangle_input.triangleArray.indexFormat  = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes  = sizeof(Face);
            triangle_input.triangleArray.numIndexTriplets    = mesh->faces.size();
            triangle_input.triangleArray.indexBuffer         = mesh->getFaceBuffer();

            if(mesh->pre_transform)
            {
                triangle_input.triangleArray.transformFormat =  OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
                triangle_input.triangleArray.preTransform        = mesh->pre_transform;
            }
            

            // ADDITIONAL SETTINGS
            // move them to mesh object
            triangle_input.triangleArray.flags         = (const uint32_t [1]) { 
                OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
            };

            // TODO: this is bad. I define the sbt records inside the sensor programs. 
            triangle_input.triangleArray.numSbtRecords = 1;
            build_inputs[idx] = triangle_input;

            // SBT data
            sbt_data_h.geometries[idx].mesh_data = mesh->sbt_data;
            sbt_data_h.geometries[idx].mesh_data.id = elem.first;
            // std::cout << "Connect GAS SBT " << idx << " -> Mesh " << elem.first << std::endl;
        } else {
            std::cout << "WARNING COULD NOT FILL GAS INPUTS" << std::endl;
        }

        idx ++;
    }

    // copy sbt
    CUDA_CHECK( cudaMemcpyAsync(
        sbt_data.geometries, 
        sbt_data_h.geometries, 
        sizeof(OptixGeomSBT) * n_build_inputs, 
        cudaMemcpyHostToDevice, m_stream->handle()) );

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

    if(!m_geom_added && !m_geom_removed && m_as)
    {
        // UPDATE
        // std::cout << "GAS - UPDATE!" << std::endl;
        accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    } else {
        // BUILD   
        // std::cout << "GAS - BUILD!" << std::endl;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    }

    m_geom_added = false;
    m_geom_removed = false;


    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                m_ctx->ref(),
                &accel_options,
                build_inputs,
                n_build_inputs, // Number of build inputs
                &gas_buffer_sizes
                ) );

    
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes) );
    
    if(!m_as)
    {
        // make new
        m_as = std::make_shared<OptixAccelerationStructure>();
        CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &m_as->buffer ),
                gas_buffer_sizes.outputSizeInBytes
                ) );
    } else {
        if(m_as->buffer_size != gas_buffer_sizes.outputSizeInBytes)
        {
            // realloc
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_as->buffer ) ) );
            CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_as->buffer ),
                    gas_buffer_sizes.outputSizeInBytes
                    ) );
        }
    }
    
    m_as->buffer_size = gas_buffer_sizes.outputSizeInBytes;
    m_as->n_elements = n_build_inputs;

    OPTIX_CHECK( optixAccelBuild(
                m_ctx->ref(),
                m_stream->handle(),                  // CUDA stream
                &accel_options,
                build_inputs,
                n_build_inputs,                  // num build inputs
                d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                m_as->buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &m_as->handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );

    m_traversable_graph_flags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    m_depth = 1;
    m_required_sbt_entries = n_build_inputs;
}

void OptixScene::buildIAS()
{
    // std::cout << "[OptixScene::buildIAS()] start." << std::endl;
    const size_t n_instances = m_geometries.size();

    // fill m_hitgroup_data
    Memory<OptixInstance, RAM> inst_h(n_instances);

    OptixSceneSBT sbt_data_h;
    cudaMallocHost(&sbt_data_h.geometries, sizeof(OptixGeomSBT) * n_instances);

    unsigned int required_sbt_entries = 0;
    unsigned int depth_ = 0;

    size_t idx = 0;
    for(auto elem : m_geometries)
    {
        unsigned int inst_id = elem.first;
        OptixInstPtr inst = std::dynamic_pointer_cast<OptixInst>(elem.second);
        inst_h[idx] = inst->data();
        inst_h[idx].instanceId = elem.first;

        required_sbt_entries = std::max(required_sbt_entries, inst->scene()->requiredSBTEntries()); 
        depth_ = std::max(depth_, inst->scene()->depth() + 1);

        sbt_data_h.geometries[idx].inst_data = inst->sbt_data;

        idx++;
    }

    // std::cout << "- COPY INSTANCE DATA" << std::endl;
    
    // COPY INSTANCES DATA
    CUdeviceptr m_inst_buffer;
    CUDA_CHECK( cudaMalloc( 
        reinterpret_cast<void**>( &m_inst_buffer ), 
        inst_h.size() * sizeof(OptixInstance) ) );

    CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( m_inst_buffer ),
                inst_h.raw(),
                inst_h.size() * sizeof(OptixInstance),
                cudaMemcpyHostToDevice,
                m_stream->handle()
                ) );

    

    // std::cout << "- COPY SBT DATA" << std::endl;
    if(n_instances > sbt_data.n_geometries)
    {
        CUDA_CHECK( cudaFree( sbt_data.geometries ) );
        CUDA_CHECK( cudaMalloc(&sbt_data.geometries, sizeof(OptixGeomSBT) * n_instances) );
    }

    sbt_data.n_geometries = n_instances;
    sbt_data.type = m_type;

    // COPY INSTANCES SBT DATA
    CUDA_CHECK( cudaMemcpyAsync(
        sbt_data.geometries,
        sbt_data_h.geometries,
        sizeof(OptixGeomSBT) * n_instances,
        cudaMemcpyHostToDevice,
        m_stream->handle()
    ) );

    // we dont need the host memory anymore
    cudaFreeHost(sbt_data_h.geometries);


    // std::cout << "- MAKE BUILD INPUT" << std::endl;
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
    
    if(!m_geom_added && !m_geom_removed && m_as)
    {
        // UPDATE
        // std::cout << "IAS - UPDATE!" << std::endl;
        ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    } else {
        // BUILD   
        // std::cout << "IAS - BUILD!" << std::endl;
        ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    }

    m_geom_added = false;
    m_geom_removed = false;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( 
        m_ctx->ref(), 
        &ias_accel_options,
        &instance_input, 
        1, 
        &ias_buffer_sizes ) );

    
    CUdeviceptr d_temp_buffer_ias;
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_ias ),
        ias_buffer_sizes.tempSizeInBytes) );

    
    if(!m_as)
    {
        // make new
        m_as = std::make_shared<OptixAccelerationStructure>();
        CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &m_as->buffer ),
                ias_buffer_sizes.outputSizeInBytes
                ) );
    } else {
        if(m_as->buffer_size != ias_buffer_sizes.outputSizeInBytes)
        {
            // realloc
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_as->buffer ) ) );
            CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_as->buffer ),
                    ias_buffer_sizes.outputSizeInBytes
                    ) );
        }
    }
    
    m_as->buffer_size = ias_buffer_sizes.outputSizeInBytes;
    m_as->n_elements = n_instances;


    OPTIX_CHECK(optixAccelBuild( 
        m_ctx->ref(), 
        m_stream->handle(), 
        &ias_accel_options, 
        &instance_input, 
        1, // num build inputs
        d_temp_buffer_ias,
        ias_buffer_sizes.tempSizeInBytes, 
        m_as->buffer,
        ias_buffer_sizes.outputSizeInBytes,
        &m_as->handle,
        nullptr, 
        0 
    ));

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_ias ) ) );

    if(depth_ < 3)
    {
        m_traversable_graph_flags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    } else {
        m_traversable_graph_flags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    }

    // if(depth_ != m_depth)
    // {
    //     std::cout << "DEPTH CHANGED! " << m_depth << " -> " << depth_ << std::endl;
    // }

    // if(m_required_sbt_entries != required_sbt_entries)
    // {
    //     std::cout << "SBT CHANGED! " << m_required_sbt_entries << " -> " << required_sbt_entries << std::endl;
    // }


    m_required_sbt_entries = required_sbt_entries;
    m_depth = depth_;

}

OptixSensorProgram OptixScene::registerSensorProgram(const OptixSimulationDataGeneric& flags)
{
    // std::cout << "REGISTER SENSOR PROGRAM" << std::endl;
    OptixSensorProgram program;
    
    // see if pipeline already exists
    auto pipe_it = m_pipelines.find(flags);
    auto sbt_it = m_sbts.find(flags);

    if(pipe_it != m_pipelines.end() && sbt_it != m_sbts.end())
    {
        program.pipeline = pipe_it->second;
        program.sbt = sbt_it->second;
    } else {

        std::cout << "[OptixScene::registerSensorProgram() Register new sensor program" << std::endl;
        auto cuda_ctx = m_ctx->getCudaContext();
        if(!cuda_ctx->isActive())
        {
            std::cout << "[OptixScene::registerSensorProgram() Need to activate map context" << std::endl;
            cuda_ctx->use();
        }

        // no pipeline found
        char log[2048]; // For error reporting from OptiX creation functions
        size_t sizeof_log = sizeof( log );

        // 1. see if raygen module already exists

        RayGenModulePtr raygen_module;
        auto raygen_it = m_sensor_raygen_modules.find(flags.model_type);
        if(raygen_it != m_sensor_raygen_modules.end())
        {
            raygen_module = raygen_it->second;
        } else {
            // std::cout << "1. make new raygen module" << std::endl;
            // make new raygen module
            raygen_module = std::make_shared<RayGenModule>();
            
            // std::cout << "- raygen_ptx_from_model_type " << std::endl;
            std::string ptx = raygen_ptx_from_model_type(flags.model_type);

            
            if(ptx.empty())
            {
                throw std::runtime_error("OptixScene could not find its PTX part");
            }

            // std::cout << "- raygen_ptx_from_model_type - done." << std::endl;

            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    #ifndef NDEBUG
            module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    #else
            module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    #endif

            module_compile_options.numPayloadTypes = 1;
            module_compile_options.payloadTypes = &m_payload_type;

            // std::cout << "- optixModuleCreateFromPTX " << std::endl;

            OPTIX_CHECK( optixModuleCreateFromPTX(
                    m_ctx->ref(),
                    &module_compile_options,
                    &m_pipeline_compile_options,
                    ptx.c_str(),
                    ptx.size(),
                    log,
                    &sizeof_log,
                    &raygen_module->module
                    ));

            // std::cout << "- optixModuleCreateFromPTX - done." << std::endl;

            
            // MAKE PROGRAM GROUPS
            // 1.1. Raygen programs
            OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = raygen_module->module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

            // std::cout << "- optixProgramGroupCreate " << std::endl;

            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        m_ctx->ref(),
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &m_program_group_options,
                        log,
                        &sizeof_log,
                        &raygen_module->prog_group
                        ) );

            // std::cout << "- optixProgramGroupCreate - done." << std::endl;

            const size_t raygen_record_size     = sizeof( RayGenModule::RayGenSbtRecord );
            CUDA_CHECK( cudaMallocHost( &raygen_module->record_h, raygen_record_size ) );
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_module->prog_group, &raygen_module->record_h[0] ) );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_module->record ), raygen_record_size ) );

            CUDA_CHECK( cudaMemcpyAsync(
                reinterpret_cast<void*>( raygen_module->record ),
                raygen_module->record_h,
                raygen_record_size,
                cudaMemcpyHostToDevice,
                m_stream->handle()
                ) );

            m_sensor_raygen_modules[flags.model_type] = raygen_module;
            // std::cout << "1. make new raygen module - done." << std::endl;
        }

        // 2. see if hit module already exists
        unsigned int bounding_id = boundingId(flags);

        HitModulePtr hit_module;
        auto hit_it = m_hit_modules.find(bounding_id);
        if(hit_it != m_hit_modules.end())
        {
            hit_module = hit_it->second;
        } else {
            // std::cout << "2. make new hit module" << std::endl;
            // need to make new hit module
            hit_module = std::make_shared<HitModule>();

            std::vector<OptixModuleCompileBoundValueEntry> options = make_bounds(flags);


            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        #ifndef NDEBUG
            // std::cout << "OPTIX_COMPILE_DEBUG_LEVEL_FULL" << std::endl;
            module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        #else
            module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
        #endif
            module_compile_options.boundValues = &options[0];
            module_compile_options.numBoundValues = options.size();

            module_compile_options.numPayloadTypes = 1;
            module_compile_options.payloadTypes = &m_payload_type;

            std::string ptx = hit_ptx();

            OPTIX_CHECK( optixModuleCreateFromPTX(
                    m_ctx->ref(),
                    &module_compile_options,
                    &m_pipeline_compile_options,
                    ptx.c_str(),
                    ptx.size(),
                    log,
                    &sizeof_log,
                    &hit_module->module
                    ));

            // MAKE PROGRAM GROUPS
            { // 2.1. Miss programs
                OptixProgramGroupDesc miss_prog_group_desc = {};

                miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
                miss_prog_group_desc.miss.module            = hit_module->module;
                miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

                OPTIX_CHECK_LOG(optixProgramGroupCreate(
                        m_ctx->ref(),
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &m_program_group_options,
                        log,
                        &sizeof_log,
                        &hit_module->prog_group_miss
                        ));
            }

            { // 2.2. Closest Hit programs
                OptixProgramGroupDesc hitgroup_prog_group_desc = {};

                hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hitgroup_prog_group_desc.hitgroup.moduleCH            = hit_module->module;
                hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
                
                OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        m_ctx->ref(),
                        &hitgroup_prog_group_desc,
                        1,   // num program groups
                        &m_program_group_options,
                        log,
                        &sizeof_log,
                        &hit_module->prog_group_hit
                        ));
            }


            // make sbt records
            { // MISS RECORDS
                const size_t n_miss_record = 1;
                hit_module->record_miss_stride = sizeof( HitModule::MissSbtRecord );
                hit_module->record_miss_count  = n_miss_record;
                const size_t miss_record_size  = hit_module->record_miss_stride * hit_module->record_miss_count;
                
                CUDA_CHECK( cudaMallocHost( &hit_module->record_miss_h, miss_record_size ) );
                for(size_t i=0; i<n_miss_record; i++)
                {
                    OPTIX_CHECK( optixSbtRecordPackHeader( hit_module->prog_group_miss, &hit_module->record_miss_h[i] ) );
                }
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hit_module->record_miss ), miss_record_size ) );

                CUDA_CHECK( cudaMemcpyAsync(
                    reinterpret_cast<void*>( hit_module->record_miss ),
                    hit_module->record_miss_h,
                    miss_record_size,
                    cudaMemcpyHostToDevice,
                    m_stream->handle()
                    ) );
            }

            { // HIT RECORDS
                const size_t n_hitgroup_records = requiredSBTEntries();   
                hit_module->record_hit_stride  = sizeof( HitModule::HitGroupSbtRecord );
                hit_module->record_hit_count   = n_hitgroup_records;
                const size_t hitgroup_record_size   = hit_module->record_hit_stride * hit_module->record_hit_count;

                CUDA_CHECK( cudaMallocHost( &hit_module->record_hit_h, hitgroup_record_size ) );
                for(size_t i=0; i<n_hitgroup_records; i++)
                {
                    OPTIX_CHECK( optixSbtRecordPackHeader( hit_module->prog_group_hit, &hit_module->record_hit_h[i] ) );
                    hit_module->record_hit_h[i].data = sbt_data;
                }
                
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hit_module->record_hit ), hitgroup_record_size ) );

                CUDA_CHECK( cudaMemcpyAsync(
                    reinterpret_cast<void*>( hit_module->record_hit ),
                    hit_module->record_hit_h,
                    hitgroup_record_size,
                    cudaMemcpyHostToDevice,
                    m_stream->handle()
                    ) );

            }

            m_hit_modules[bounding_id] = hit_module;
            // std::cout << "2. make new hit module - done." << std::endl;
        }

        // 3. make sbt
        auto sbt_it = m_sbts.find(flags);
        OptixSBTPtr sbt_;
        if(sbt_it != m_sbts.end())
        {
            sbt_ = sbt_it->second;
        } else {
            sbt_ = std::make_shared<OptixSBT>();
            OptixShaderBindingTable& sbt = sbt_->sbt;

            sbt.raygenRecord = raygen_module->record;
            
            sbt.missRecordBase = hit_module->record_miss;
            sbt.missRecordStrideInBytes = hit_module->record_miss_stride;
            sbt.missRecordCount = hit_module->record_miss_count;

            sbt.hitgroupRecordBase = hit_module->record_hit;
            sbt.hitgroupRecordStrideInBytes = hit_module->record_hit_stride;
            sbt.hitgroupRecordCount = hit_module->record_hit_count;

            m_sbts[flags] = sbt_;
        }
        program.sbt = sbt_;

        
        { // 4. link pipeline. independend of SBT
            OptixSensorPipelinePtr pipeline_ = std::make_shared<OptixSensorPipeline>();

            // traverse depth = 2 for ias + gas
            uint32_t          max_traversable_depth = m_depth;
            const uint32_t    max_trace_depth  = 1; // TODO: 31 is maximum. Set this dynamically?

            OptixProgramGroup program_groups[] = { 
                raygen_module->prog_group, 
                hit_module->prog_group_miss, 
                hit_module->prog_group_hit
            };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth          = max_trace_depth;
            #ifndef NDEBUG
                pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            #else
                pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
            #endif
            OPTIX_CHECK_LOG( optixPipelineCreate(
                m_ctx->ref(),
                    &m_pipeline_compile_options,
                    &pipeline_link_options,
                    program_groups,
                    sizeof(program_groups) / sizeof(program_groups[0]),
                    log,
                    &sizeof_log,
                    &pipeline_->pipeline
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
                                                        &direct_callable_stack_size_from_state, 
                                                        &continuation_stack_size ) );
            OPTIX_CHECK( optixPipelineSetStackSize( pipeline_->pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state, continuation_stack_size,
                                                    max_traversable_depth  // maxTraversableDepth
                                                    ) );

            m_pipelines[flags] = pipeline_;
            program.pipeline = pipeline_;
        }
    }

    // std::cout << "REGISTER SENSOR PROGRAM - finished." << std::endl;
    return program;
}

void OptixScene::updateSBT()
{
    const size_t n_hitgroups_required = requiredSBTEntries();

    for(auto elem : m_sbts)
    {
        unsigned int bounding_id = boundingId(elem.first);
        OptixSBTPtr sbt = elem.second;
        
        auto hit_it = m_hit_modules.find(bounding_id);

        if(hit_it != m_hit_modules.end())
        {
            HitModulePtr hit_module = hit_it->second;

            if(n_hitgroups_required > hit_module->record_hit_count)
            {
                // std::cout << "UPDATE SBT SIZE!" << std::endl;
                CUDA_CHECK( cudaFreeHost( hit_module->record_hit_h ) );
                CUDA_CHECK( cudaMallocHost( &hit_module->record_hit_h, n_hitgroups_required * hit_module->record_hit_stride ) );

                for(size_t i=0; i<n_hitgroups_required; i++)
                {
                    OPTIX_CHECK( optixSbtRecordPackHeader( hit_module->prog_group_hit, &hit_module->record_hit_h[i] ) );
                }

                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( hit_module->record_hit ) ) );
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hit_module->record_hit ), n_hitgroups_required * hit_module->record_hit_stride ) );
                
                hit_module->record_hit_count = n_hitgroups_required;
            }

            for(size_t i=0; i<hit_module->record_hit_count; i++)
            {
                hit_module->record_hit_h[i].data = sbt_data;
            }

            CUDA_CHECK( cudaMemcpyAsync(
                    reinterpret_cast<void*>( hit_module->record_hit ),
                    hit_module->record_hit_h,
                    hit_module->record_hit_count * hit_module->record_hit_stride,
                    cudaMemcpyHostToDevice,
                    m_stream->handle()
                    ) );

            sbt->sbt.hitgroupRecordBase = hit_module->record_hit;
            sbt->sbt.hitgroupRecordStrideInBytes = hit_module->record_hit_stride;
            sbt->sbt.hitgroupRecordCount = hit_module->record_hit_count;

        } else {
            std::cout << "[OptixScene::updateSBT()] ERROR - cannot find hit module of sbt" << std::endl;
        }
    }
}

OptixScenePtr make_optix_scene(
    const aiScene* ascene, 
    OptixContextPtr context)
{
    OptixScenePtr scene = std::make_shared<OptixScene>(context);

    std::map<unsigned int, OptixMeshPtr> meshes;
    // 1. meshes
    // std::cout << "[make_optix_scene()] Loading Meshes..." << std::endl;

    for(size_t i=0; i<ascene->mNumMeshes; i++)
    {
        // std::cout << "Make Mesh " << i+1 << "/" << ascene->mNumMeshes << std::endl;
        const aiMesh* amesh = ascene->mMeshes[i];

        if(amesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)
        {
            // triangle mesh
            OptixMeshPtr mesh = std::make_shared<OptixMesh>(amesh);
            mesh->commit();
            meshes[i] = mesh;
            // std::cout << "Mesh " << i << "(" << mesh->name << ") added." << std::endl;
        } else {
            std::cout << "[ make_optix_scene(aiScene) ] WARNING: Could not construct geometry " << i << " prim type " << amesh->mPrimitiveTypes << " not supported yet. Skipping." << std::endl;
        }
    }

    // 2. instances (if available)
    std::unordered_set<OptixGeometryPtr> instanciated_meshes;
    const aiNode* root_node = ascene->mRootNode;
    std::vector<const aiNode*> mesh_nodes = get_nodes_with_meshes(root_node);

    // std::cout << "[make_embree_scene()] Loading Instances..." << std::endl;
    for(size_t i=0; i<mesh_nodes.size(); i++)
    {
        const aiNode* node = mesh_nodes[i];
        
        Matrix4x4 M = global_transform(node);
        Transform T;
        Vector3 scale;
        decompose(M, T, scale);

        OptixScenePtr mesh_scene = std::make_shared<OptixScene>();

        for(unsigned int i = 0; i<node->mNumMeshes; i++)
        {
            unsigned int mesh_id = node->mMeshes[i];
            auto mesh_it = meshes.find(mesh_id);
            if(mesh_it != meshes.end())
            {
                // mesh found
                OptixMeshPtr mesh = mesh_it->second;
                instanciated_meshes.insert(mesh);
                mesh_scene->add(mesh);
                mesh_scene->commit();
            } else {
                std::cout << "[make_optix_scene()] WARNING: could not find mesh_id " 
                    << mesh_id << " in meshes during instantiation" << std::endl;
            }
        }

        mesh_scene->commit();

        // std::cout << "--- mesh added to mesh_scene" << std::endl;
        OptixInstPtr mesh_instance = std::make_shared<OptixInst>();
        mesh_instance->set(mesh_scene);
        mesh_instance->name = node->mName.C_Str();
        mesh_instance->setTransform(T);
        mesh_instance->setScale(scale);
        mesh_instance->apply();
        mesh_instance->commit();
        // std::cout << "--- mesh_instance created" << std::endl;
        unsigned int inst_id = scene->add(mesh_instance);
        // std::cout << "Instance " << inst_id << " (" << mesh_instance->name << ") added" << std::endl;
    }

    // if(scene->type() != OptixSceneType::INSTANCES)
    // {
    //     std::cout << "add meshes that are not instanciated as geometry" << std::endl;
    // } else {
    //     std::cout << "add meshes that are not instanciated as instance" << std::endl;
    // }

    // ADD MESHES THAT ARE NOT INSTANCIATED
    for(auto elem : meshes)
    {
        auto mesh = elem.second;
        if(instanciated_meshes.find(mesh) == instanciated_meshes.end())
        {
            // mesh was never instanciated. add to scene
            if(scene->type() != OptixSceneType::INSTANCES)
            {
                scene->add(mesh);
            } else {
                // mesh->instantiate();
                OptixInstPtr geom_inst = mesh->instantiate();
                geom_inst->apply();
                geom_inst->commit();
                scene->add(geom_inst);
            }
        }
    }

    return scene;
}



} // namespace rmagine