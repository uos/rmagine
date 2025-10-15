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



int main(int argc, char** argv)
{
    std::cout << "Main start." << std::endl;

    // minimum 2 arguments
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " mesh_file" << std::endl;
        return EXIT_SUCCESS;
    }
    std::string path_to_mesh = argv[1];
    std::cout << "Inputs: " << std::endl;
    std::cout << "- mesh: " << path_to_mesh << std::endl;


    //create map
    std::cout << "Creating map." << std::endl;
    VulkanMapPtr map = import_vulkan_map(path_to_mesh);


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
    tsb.setIdentity();
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
    for(size_t i = 0; i < tbm_ram.size(); i++)
    {
        tbm_ram[i] = tsb;
    }
    Memory<Transform, DEVICE_LOCAL_VULKAN> tbm(tbm_ram.size());
    tbm = tbm_ram;


    //bundle
    using ResultT = Bundle<
        Ranges<DEVICE_LOCAL_VULKAN> 
        // ,Hits<DEVICE_LOCAL_VULKAN>
        // ,Points<DEVICE_LOCAL_VULKAN>
        // ,Normals<DEVICE_LOCAL_VULKAN>
        // ,FaceIds<DEVICE_LOCAL_VULKAN>   //primitive ids
        // ,GeomIds<DEVICE_LOCAL_VULKAN>   //geometry ids
        // ,ObjectIds<DEVICE_LOCAL_VULKAN> //instance ids
    >;

    //allocate sphere resultsbuffer
    std::cout << "Creating sphere results." << std::endl;
    Memory<float, RAM> sphereRanges_ram(tbm_ram.size()*sphereSensor.phi.size*sphereSensor.theta.size);
    ResultT res;
    res.ranges.resize(sphereRanges_ram.size());
    

    //simulate sphere
    std::cout << "Simulating sphere..." << std::endl;
    sim_gpu_sphere->simulate(tbm, res);


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
        sim_gpu_sphere->simulate(tbm, res);
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
    std::cout << "Result: " << velos_per_second_mean << " velos/s\n" << std::endl;

    return EXIT_SUCCESS;
}
