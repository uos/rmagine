// Core rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>

// Vulkan rmagine includes
#include <rmagine/simulation/SphereSimulatorVulkan.hpp>
#include <rmagine/map/VulkanMap.hpp>
#include <rmagine/types/MemoryVulkan.hpp>
#include <rmagine/map/vulkan/vulkan_shapes.hpp>


using namespace rmagine;



VulkanMapPtr make_sphere_map(unsigned int num_long, unsigned int num_lat)
{
    VulkanScenePtr scene = std::make_shared<VulkanScene>();

    VulkanMeshPtr mesh = std::make_shared<VulkanSphere>(num_long, num_lat);
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    std::cout << "Num Faces: " << mesh->faces.size() << std::endl;

    return std::make_shared<VulkanMap>(scene);
}



size_t reps = 100;

size_t num_maps = 10;
size_t map_param = 50;

size_t num_tbms = 10;

int main(int argc, char** argv)
{
    VulkanMapPtr map = make_sphere_map(map_param*num_maps, map_param*num_maps);

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



    Transform tsb = Transform();
    tsb.setIdentity();
    Memory<Transform, RAM> tsb_ram(1);
    tsb_ram[0] = tsb;



    SphereSimulatorVulkanPtr sim_gpu_sphere = std::make_shared<SphereSimulatorVulkan>(map);
    sim_gpu_sphere->setModel(sphereSensor_ram);
    sim_gpu_sphere->setTsb(tsb_ram);



    Memory<Transform, RAM> tbm_ram(1000*num_tbms);
    for(size_t i = 0; i < tbm_ram.size(); i++)
    {
        //TODO: fill with random poses
        tbm_ram[i] = tsb;
    }
    Memory<Transform, DEVICE_LOCAL_VULKAN> tbm(tbm_ram.size());
    tbm = tbm_ram;



    using ResultT = Bundle<
        Ranges<DEVICE_LOCAL_VULKAN> 
        // ,Hits<DEVICE_LOCAL_VULKAN>
        // ,Points<DEVICE_LOCAL_VULKAN>
        // ,Normals<DEVICE_LOCAL_VULKAN>
        // ,FaceIds<DEVICE_LOCAL_VULKAN>   //primitive ids
        // ,GeomIds<DEVICE_LOCAL_VULKAN>   //geometry ids
        // ,ObjectIds<DEVICE_LOCAL_VULKAN> //instance ids
    >;

    ResultT res;
    res.ranges.resize(tbm_ram.size()*sphereSensor.phi.size*sphereSensor.theta.size);



    rmagine::StopWatch sw;
    std::cout << "\n" << std::endl;



    for(size_t i = 1; i <= num_tbms; i++)
    {
        Memory<Transform, RAM> tbm2_ram(1000*i);
        for(size_t j = 0; j < tbm2_ram.size(); j++)
        {
            //TODO: fill with random poses
            tbm2_ram[j] = tsb;
        }
        Memory<Transform, DEVICE_LOCAL_VULKAN> tbm2(tbm2_ram.size());
        tbm2 = tbm2_ram;
        std::cout << "Num Poses: " << tbm2.size() << std::endl;


        ResultT res2;
        res2.ranges.resize(tbm2.size()*sphereSensor.phi.size*sphereSensor.theta.size);


        //prebuild
        sim_gpu_sphere->simulate(tbm2, res2);


        //measure
        double elapsed = 0.0;
        double elapsed_total = 0.0;
        std::cout << "-- Starting Measurement --" << std::endl;
        for(size_t j = 0; j < reps; j++)
        {
            double run = static_cast<double>(j) + 1.0;

            sw();
            sim_gpu_sphere->simulate(tbm2, res2);
            double elapsed = sw();
            elapsed_total += elapsed;
            
            std::cout 
            << std::fixed
            << "[Elapsed: " << elapsed << "; "
            << "Elapsed Total: " << elapsed_total << "; "
            << "Elapsed Average: " << elapsed_total/(run) << "]" << ((j+1==reps) ? "" : "\r");
            std::cout.flush();
        }
        std::cout << "\n" << std::endl;
    }


    std::cout << "\n" << std::endl;


    for(size_t i = 1; i <= num_maps; i++)
    {
        VulkanMapPtr map2 = make_sphere_map(map_param*i, map_param*i);
        sim_gpu_sphere->setMap(map2);


        //prebuild
        sim_gpu_sphere->simulate(tbm, res);


        //measure
        double elapsed = 0.0;
        double elapsed_total = 0.0;
        std::cout << "-- Starting Measurement --" << std::endl;
        for(size_t j = 0; j < reps; j++)
        {
            double run = static_cast<double>(j) + 1.0;

            sw();
            sim_gpu_sphere->simulate(tbm, res);
            elapsed = sw();
            elapsed_total += elapsed;
            
            std::cout 
            << std::fixed
            << "[Elapsed: " << elapsed << "; "
            << "Elapsed Total: " << elapsed_total << "; "
            << "Elapsed Average: " << elapsed_total/(run) << "]" << ((j+1==reps) ? "" : "\r");
            std::cout.flush();
        }
        std::cout << "\n" << std::endl;
    }

    std::cout << "\nFinished..." << std::endl;

    return EXIT_SUCCESS;
}
