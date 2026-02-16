#include <iostream>

// Core rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>


// CPU - Embree
#if defined WITH_EMBREE
#include <rmagine/simulation/SphereSimulatorEmbree.hpp>
#include <rmagine/simulation/PinholeSimulatorEmbree.hpp>
#endif

// GPU - Optix
#if defined WITH_OPTIX
#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/types/MemoryCuda.hpp>
#endif

#include "benchmarks/velodyne_benchmark.hpp"


using namespace rmagine;

// Memory<LiDARModel, RAM> velodyne_model()
// {
//     Memory<LiDARModel, RAM> model(1);
//     model->theta.min = -M_PI;
//     model->theta.inc = 0.4 * M_PI / 180.0;
//     model->theta.size = 900;

//     model->phi.min = -15.0 * M_PI / 180.0;
//     model->phi.inc = 2.0 * M_PI / 180.0;
//     model->phi.size = 16;
    
//     model->range.min = 0.1;
//     model->range.max = 130.0;
//     return model;
// }

int main(int argc, char** argv)
{
    std::cout << "Rmagine Benchmark";
    
    int device_id = 0;
    std::string device;

    #if defined WITH_EMBREE
    std::cout << " CPU (Embree)";
    device = "cpu";
    #elif defined WITH_OPTIX
    std::cout << " GPU (OptiX)";
    device = "gpu";
    #else
    Either Embree or OptiX must be defined // compile time error
    #endif
    
    std::cout << std::endl;

    // Total runtime of the Benchmark in seconds
    double benchmark_duration = 10.0;
    // Poses to check per call
    size_t Nposes = 10 * 1024;

    // minimum 2 arguments
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " mesh_file" << std::endl;
        return 0;
    }

    std::string path_to_mesh = argv[1];
    // std::string device = argv[2];
    
    std::cout << "Inputs: " << std::endl;
    std::cout << "- mesh: " << path_to_mesh << std::endl;

    StopWatch sw;
    double elapsed;
    double elapsed_total = 0.0;

    Memory<LiDARModel, RAM> model = velodyne_model();

    std::cout << "Unit: 1 Velodyne scan (velo) = " << model[0].size() << " Rays" << std::endl;

    if(device == "cpu")
    {
        #if defined WITH_EMBREE
        
        // Load mesh
        EmbreeMapPtr cpu_mesh = import_embree_map(path_to_mesh);
        auto cpu_sim = std::make_shared<SphereSimulatorEmbree>(cpu_mesh);

        velodyne_benchmark(
            VelodyneBenchmarkConfig{
              .duration = benchmark_duration,
              .n_poses = Nposes
            },
            cpu_sim
        );        

        // clean up
        #else // WITH_EMBREE

        std::cout << "cpu benchmark not possible. Compile with Embree support." << std::endl;

        #endif

    } else if(device == "gpu") {
        #if defined WITH_OPTIX

        OptixMapPtr gpu_mesh = import_optix_map(path_to_mesh);
        SphereSimulatorOptixPtr gpu_sim = std::make_shared<SphereSimulatorOptix>(gpu_mesh);

        velodyne_benchmark(
            VelodyneBenchmarkConfig{
              .duration = benchmark_duration,
              .n_poses = Nposes
            },
            gpu_sim
        );

        #else // WITH_OPTIX
            std::cout << "gpu benchmark not possible. Compile with OptiX support." << std::endl;
        #endif
    } else {
        std::cout << "Device " << device << " unknown" << std::endl;
    }

    return 0;
}
