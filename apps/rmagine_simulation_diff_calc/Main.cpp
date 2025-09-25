// Core rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>

// Vulkan rmagine includes
#include <rmagine/simulation/SphereSimulatorVulkan.hpp>
#include <rmagine/map/VulkanMap.hpp>
#include <rmagine/types/MemoryVulkan.hpp>
// #include <rmagine/simulation/SphereSimulatorOptix.hpp>
// #include <rmagine/map/OptixMap.hpp>
// #include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/simulation/SphereSimulatorEmbree.hpp>
#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/types/Memory.hpp>


using namespace rmagine;



int main(int argc, char** argv)
{
    std::cout << "Main start." << std::endl;



    size_t nPoses = 1;

    // minimum 2 arguments
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " mesh_file" << std::endl;
        return EXIT_SUCCESS;
    }
    std::string path_to_mesh = argv[1];
    std::cout << "Inputs: " << std::endl;
    std::cout << "- mesh: " << path_to_mesh << std::endl;



    // Define one Transform Sensor to Base
    Memory<Transform, RAM> tsb_ram(1);
    tsb_ram->R.x = 0.0;
    tsb_ram->R.y = 0.0;
    tsb_ram->R.z = 0.0;
    tsb_ram->R.w = 1.0;
    tsb_ram->t.x = 0.0;
    tsb_ram->t.y = 0.0;
    tsb_ram->t.z = 0.0;



    // Define Transforms Base to Map (Poses)
    Memory<Transform, RAM> tbm_ram(nPoses);
    for(size_t i=0; i<tbm_ram.size(); i++)
    {
        tbm_ram[i] = tsb_ram[0];
    }
    // Memory<Transform, RAM_CUDA> tbm_ramCuda(nPoses);
    // for(size_t i=0; i<tbm_ramCuda.size(); i++)
    // {
    //     tbm_ramCuda[i] = tsb_ram[0];
    // }
    // Memory<Transform, VRAM_CUDA> tbm_vramCuda(nPoses);
    // tbm_vramCuda = tbm_ramCuda;
    Memory<Transform, DEVICE_LOCAL_VULKAN> tbm_vulkan(nPoses);
    tbm_vulkan = tbm_ram;



    // Load mesh
    EmbreeMapPtr map_embree = import_embree_map(path_to_mesh);
    // OptixMapPtr map_optix = import_optix_map(path_to_mesh);
    VulkanMapPtr map_vulkan = import_vulkan_map(path_to_mesh);



    // Sensor
    Memory<LiDARModel, RAM> model_ram(1);
    model_ram->theta.min = -M_PI;
    model_ram->theta.inc = 0.4 * M_PI / 180.0;
    model_ram->theta.size = 900;
    model_ram->phi.min = -15.0 * M_PI / 180.0;
    model_ram->phi.inc = 2.0 * M_PI / 180.0;
    model_ram->phi.size = 16;
    model_ram->range.min = 0.1;
    model_ram->range.max = 130.0;



    // Results
    using ResultEmbreeT = Bundle<
        // Hits<RAM>,
        // Points<RAM>,
        // Normals<RAM>,
        // FaceIds<RAM>,
        // GeomIds<RAM>,
        // ObjectIds<RAM>
        Ranges<RAM>
    >;
    ResultEmbreeT res_ram;
    res_ram.ranges.resize(tbm_ram.size() * model_ram->size());

    // using ResultOptixT = Bundle<
    //     // Hits<VRAM_CUDA>,
    //     // Points<VRAM_CUDA>,
    //     // Normals<VRAM_CUDA>,
    //     // FaceIds<VRAM_CUDA>,
    //     // GeomIds<VRAM_CUDA>,
    //     // ObjectIds<VRAM_CUDA>,
    //     Ranges<VRAM_CUDA>
    // >;
    // ResultOptixT res_vramCuda;
    // res_vramCuda.ranges.resize(tbm_vramCuda.size() * model_ram->size());

    using ResultVulkanT = Bundle<
        // Hits<DEVICE_LOCAL_VULKAN>,
        // Points<DEVICE_LOCAL_VULKAN>,
        // Normals<DEVICE_LOCAL_VULKAN>,
        // FaceIds<DEVICE_LOCAL_VULKAN>,   //primitive ids
        // GeomIds<DEVICE_LOCAL_VULKAN>,   //geometry ids
        // ObjectIds<DEVICE_LOCAL_VULKAN>,  //instance ids
        Ranges<DEVICE_LOCAL_VULKAN>
    >;
    ResultVulkanT res_vulkan;
    res_vulkan.ranges.resize(tbm_vulkan.size() * model_ram->size());



    // Simulators
    SphereSimulatorEmbreePtr sim_embree = std::make_shared<SphereSimulatorEmbree>(map_embree);
    sim_embree->setModel(model_ram);
    sim_embree->setTsb(tsb_ram);
    // SphereSimulatorOptixPtr sim_optix = std::make_shared<SphereSimulatorOptix>(map_optix);
    // sim_optix->setModel(model_ram);
    // sim_optix->setTsb(tsb_ram);
    SphereSimulatorVulkanPtr sim_vulkan = std::make_shared<SphereSimulatorVulkan>(map_vulkan);
    sim_vulkan->setModel(model_ram);
    sim_vulkan->setTsb(tsb_ram);



    // Simulate
    sim_embree->simulate(tbm_ram, res_ram);
    // sim_optix->simulate(tbm_vramCuda, res_vramCuda);
    sim_vulkan->simulate(tbm_vulkan, res_vulkan);



    // Evaluate results
    Memory<float, RAM> ranges_embree = res_ram.ranges;
    // Memory<float, RAM> ranges_optix = res_vramCuda.ranges;
    Memory<float, RAM> ranges_vulkan = res_vulkan.ranges;

    float allowed_error = 0.0001;

    float max_error_diff_embree_vulkan = 0.0;
    uint64_t count_errors_embree_vulkan = 0;
    for(size_t i = 0; i < ranges_embree.size(); i++)
    {
        float error = std::fabs(ranges_embree[i] - ranges_vulkan[i]);

        if(error > allowed_error)                                              
        {
            count_errors_embree_vulkan++;
        }
        if(error > max_error_diff_embree_vulkan)
        {
            max_error_diff_embree_vulkan = error;
        }
    }

    // float max_error_diff_optix_vulkan = 0.0;
    // uint64_t count_errors_optix_vulkan = 0;
    // for(size_t i = 0; i < ranges_optix.size(); i++)
    // {
    //     float error = std::fabs(ranges_optix[i] - ranges_vulkan[i]);
    //
    //     if(error > allowed_error)                                              
    //     {
    //         count_errors_optix_vulkan++;
    //     }
    //     if(error > max_error_diff_optix_vulkan)
    //     {
    //         max_error_diff_optix_vulkan = error;
    //     }
    // }

    // float max_error_diff_embree_optix = 0.0;
    // uint64_t count_errors_embree_optix = 0;
    // for(size_t i = 0; i < ranges_embree.size(); i++)
    // {
    //     float error = std::fabs(ranges_embree[i] - ranges_optix[i]);
    //
    //     if(error > allowed_error)                                              
    //     {
    //         count_errors_embree_optix++;
    //     }
    //     if(error > max_error_diff_embree_optix)
    //     {
    //         max_error_diff_embree_optix = error;
    //     }
    // }

    std::cout << "count_errors_embree_vulkan: " << count_errors_embree_vulkan << std::endl;
    // std::cout << "count_errors_optix_vulkan: " << count_errors_optix_vulkan << std::endl;
    // std::cout << "count_errors_embree_optix: " << count_errors_embree_optix << std::endl;

    std::cout << "max_error_diff_embree_vulkan: " << max_error_diff_embree_vulkan << std::endl;
    // std::cout << "max_error_diff_optix_vulkan: " << max_error_diff_optix_vulkan << std::endl;
    // std::cout << "max_error_diff_embree_optix: " << max_error_diff_embree_optix << std::endl;



    std::cout << "Main end." << std::endl;
    return EXIT_SUCCESS;
}
