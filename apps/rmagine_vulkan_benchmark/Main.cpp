// Core rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>

// Vulkan rmagine includes
#include <rmagine/simulation/SphereSimulatorVulkan.hpp>
#include <rmagine/map/VulkanMap.hpp>
#include <rmagine/types/MemoryVulkan.hpp>


using namespace rmagine;



int main()
{
    std::cout << "Main start." << std::endl;

    //mapdata
    Memory<float, RAM> vertexMem_ram(9);
    Memory<uint32_t, RAM> indexMem_ram(3);
    //Vertecies
    // vertexMem_ram[ 0] =  20.0f; vertexMem_ram[ 1] = -10.0f; vertexMem_ram[ 2] = -10.0f;
    // vertexMem_ram[ 3] =  20.0f; vertexMem_ram[ 4] =  10.0f; vertexMem_ram[ 5] = -10.0f;
    // vertexMem_ram[ 6] =  20.0f; vertexMem_ram[ 7] =   0.0f; vertexMem_ram[ 8] =  10.0f;
    vertexMem_ram[ 0] =  0.0f; vertexMem_ram[ 1] =  0.0f; vertexMem_ram[ 2] =  0.0f;
    vertexMem_ram[ 3] =  1.0f; vertexMem_ram[ 4] =  0.0f; vertexMem_ram[ 5] =  0.0f;
    vertexMem_ram[ 6] =  0.0f; vertexMem_ram[ 7] =  1.0f; vertexMem_ram[ 8] =  0.0f;
    //Indicies
    indexMem_ram[ 0] = 0; indexMem_ram[ 1] = 1; indexMem_ram[ 2] = 2;
    //logging
    uint32_t numVerticies = vertexMem_ram.size()/3;
    uint32_t numTriangles = indexMem_ram.size()/3;
    std::cout << "Using Mesh with " << numVerticies << " Verticies & " << numTriangles << " Triangles." << std::endl;


    //create map
    std::cout << "Creating map main." << std::endl;
    VulkanMapPtr map = import_vulkan_map(vertexMem_ram, indexMem_ram);


    //allocate sphere sensorbuffer
    std::cout << "Creating sphere sensor." << std::endl;
    SphericalModel sphereSensor = SphericalModel();
    sphereSensor.theta.min = -M_PI;
    sphereSensor.theta.inc = 0.4 * M_PI / 180.0;
    sphereSensor.theta.size = 900;
    sphereSensor.phi.min = -15.0 * M_PI / 180.0;
    sphereSensor.phi.inc = 2.0 * M_PI / 180.0;
    sphereSensor.phi.size = 16;
    sphereSensor.range.min = 0.1;
    sphereSensor.range.max = 130.0;
    Memory<SphericalModel, RAM> sphereSensor_ram(1);
    sphereSensor_ram[0] = sphereSensor;

    std::cout << "Unit: 1 Velodyne scan (velo) = " << sphereSensor.theta.size*sphereSensor.phi.size << " Rays" << std::endl;


    //allocate transformbuffer
    std::cout << "Creating transform." << std::endl;
    Transform tsb = Transform();
    Memory<Transform, RAM> tsb_ram(1);
    tsb_ram[0] = tsb;


    //create sphere simulator
    std::cout << "Creating sphere sim." << std::endl;
    SphereSimulatorVulkanPtr sim_gpu_sphere = std::make_shared<SphereSimulatorVulkan>(map);
    sim_gpu_sphere->setModel(sphereSensor_ram);
    sim_gpu_sphere->setTsb(tsb_ram);


    //allocate transformsbuffer 
    std::cout << "Creating transforms." << std::endl;
    Memory<Transform, RAM> tbm_ram(1024*10);
    Memory<Transform, VULKAN_DEVICE_LOCAL> tbm_device(tbm_ram.size());
    tbm_device = tbm_ram;


    //bundle
    using ResultT = Bundle<
        Ranges<VULKAN_DEVICE_LOCAL> 
        // ,Hits<VULKAN_DEVICE_LOCAL>
        // ,Points<VULKAN_DEVICE_LOCAL>
        // ,Normals<VULKAN_DEVICE_LOCAL>
        // ,FaceIds<VULKAN_DEVICE_LOCAL>   //primitive ids
        // ,GeomIds<VULKAN_DEVICE_LOCAL>   //geometry ids
        // ,ObjectIds<VULKAN_DEVICE_LOCAL> //instance ids
    >;

    //allocate sphere resultsbuffer
    std::cout << "Creating sphere results." << std::endl;
    Memory<float, RAM> sphereRanges_ram(tbm_ram.size()*sphereSensor.phi.size*sphereSensor.theta.size);
    ResultT res;
    res.ranges.resize(sphereRanges_ram.size());
    

    //simulate sphere
    std::cout << "Simulating sphere..." << std::endl;
    sim_gpu_sphere->simulate(tbm_device, res);


    //get sphere results
    sphereRanges_ram = res.ranges;
    std::cout << "\nSphere results:" << std::endl;
    
    if(sphereRanges_ram[0] <= sphereSensor.range.max)
        std::cout << "1" << std::endl;
    else
        std::cout << "0" << std::endl;



    std::cout << "-- Starting Benchmark --" << std::endl;

    rmagine::StopWatch sw;
    // Total runtime of the Benchmark in seconds
    double benchmark_duration = 10.0;
    double elapsed = 0.0;
    double elapsed_total = 0.0;
    double velos_per_second_mean = 0.0;

    int run = 0;
    while(elapsed_total < benchmark_duration)
    {
        double n_dbl = static_cast<double>(run) + 1.0;
        // Simulate
        sw();
        sim_gpu_sphere->simulate(tbm_device, res);
        elapsed = sw();
        elapsed_total += elapsed;
        double velos_per_second = static_cast<double>(tbm_ram.size()) / elapsed;
        velos_per_second_mean = (n_dbl - 1.0)/(n_dbl) * velos_per_second_mean + (1.0 / n_dbl) * velos_per_second; 
        
        std::cout 
        << std::fixed
        << "[ " << int((elapsed_total / benchmark_duration)*100.0) << "%" << " - " 
        << velos_per_second << " velos/s" 
        << ", mean: " << velos_per_second_mean << " velos/s] \r";
        std::cout.flush();

        run++;
    }

    std::cout << std::endl;
    std::cout << "Result: " << velos_per_second_mean << " velos/s" << std::endl;

    return EXIT_SUCCESS;
}
