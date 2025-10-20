#include <random>

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




std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<float> dist(0, 1);

Transform randomTransform()
{
    Transform tf;
    tf.R = {dist(e2), dist(e2), dist(e2), dist(e2)};
    tf.R.normalizeInplace();
    tf.t = {dist(e2), dist(e2), dist(e2)};
    tf.t.normalizeInplace();
}

void fillWithRandomTfs(Memory<Transform, RAM>& tbm_ram)
{
    for(size_t i = 0; i < tbm_ram.size(); i++)
    {
        tbm_ram[i] = randomTransform();
    }
}



void printAsPoints(std::vector<double>& results)
{
    for(size_t i = 0; i < results.size(); i++)
    {
        std::cout << "("<< i+1 << ", " << results[i] << "),";
    }
    std::cout << std::endl;
}



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
size_t map_param = 100;

size_t num_tbms = 10;
size_t tbm_param = 1000;

int main(int argc, char** argv)
{
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


    SphereSimulatorVulkanPtr sim_gpu_sphere = std::make_shared<SphereSimulatorVulkan>();
    sim_gpu_sphere->setModel(sphereSensor_ram);
    sim_gpu_sphere->setTsb(tsb_ram);


    using ResultT = Bundle<
        Ranges<DEVICE_LOCAL_VULKAN> 
        // ,Hits<DEVICE_LOCAL_VULKAN>
        // ,Points<DEVICE_LOCAL_VULKAN>
        // ,Normals<DEVICE_LOCAL_VULKAN>
        // ,FaceIds<DEVICE_LOCAL_VULKAN>   //primitive ids
        // ,GeomIds<DEVICE_LOCAL_VULKAN>   //geometry ids
        // ,ObjectIds<DEVICE_LOCAL_VULKAN> //instance ids
    >;


    rmagine::StopWatch sw;
    std::cout << "\n" << std::endl;



    //measure different num of sensors
    {
        VulkanMapPtr map = make_sphere_map(map_param*num_maps, map_param*num_maps);
        sim_gpu_sphere->setMap(map);


        std::vector<double> results;
        for(size_t i = 1; i <= num_tbms; i++)
        {
            Memory<Transform, RAM> tbm_ram(tbm_param*i);
            for(size_t j = 0; j < tbm_ram.size(); j++)
            {
                tbm_ram[j] = tsb;
            }
            Memory<Transform, DEVICE_LOCAL_VULKAN> tbm(tbm_ram.size());
            tbm = tbm_ram;
            std::cout << "Num Poses: " << tbm.size() << std::endl;


            ResultT res2;
            res2.ranges.resize(tbm.size()*sphereSensor.phi.size*sphereSensor.theta.size);


            //prebuild
            sim_gpu_sphere->simulate(tbm, res2);


            //measure
            double elapsed = 0.0;
            double elapsed_total = 0.0;
            std::cout << "-- Starting Measurement --" << std::endl;
            for(size_t j = 1; j <= reps; j++)
            {
                fillWithRandomTfs(tbm_ram);
                tbm = tbm_ram;

                sw();
                sim_gpu_sphere->simulate(tbm, res2);
                double elapsed = sw();
                elapsed_total += elapsed;
                
                std::cout 
                << std::fixed
                << "[Elapsed: " << elapsed << "; "
                << "Elapsed Total: " << elapsed_total << "; "
                << "Elapsed Average: " << elapsed_total/(static_cast<double>(j)) << "]" << ((j==reps) ? "" : "\r");
                std::cout.flush();
            }
            results.push_back(elapsed_total/(static_cast<double>(reps)));
            std::cout << "\n" << std::endl;
        }
        printAsPoints(results);
    }


    std::cout << "\n" << std::endl;


    //measure different num of faces
    {
        Memory<Transform, RAM> tbm_ram(tbm_param*num_tbms);
        for(size_t i = 0; i < tbm_ram.size(); i++)
        {
            tbm_ram[i] = tsb;
        }
        Memory<Transform, DEVICE_LOCAL_VULKAN> tbm(tbm_ram.size());
        tbm = tbm_ram;


        ResultT res;
        res.ranges.resize(tbm_ram.size()*sphereSensor.phi.size*sphereSensor.theta.size);


        std::vector<double> results;
        for(size_t i = 1; i <= num_maps; i++)
        {
            VulkanMapPtr map = make_sphere_map(map_param*i, map_param*i);
            sim_gpu_sphere->setMap(map);


            //prebuild
            sim_gpu_sphere->simulate(tbm, res);


            //measure
            double elapsed = 0.0;
            double elapsed_total = 0.0;
            std::cout << "-- Starting Measurement --" << std::endl;
            for(size_t j = 1; j <= reps; j++)
            {
                fillWithRandomTfs(tbm_ram);
                tbm = tbm_ram;

                sw();
                sim_gpu_sphere->simulate(tbm, res);
                elapsed = sw();
                elapsed_total += elapsed;
                
                std::cout 
                << std::fixed
                << "[Elapsed: " << elapsed << "; "
                << "Elapsed Total: " << elapsed_total << "; "
                << "Elapsed Average: " << elapsed_total/(static_cast<double>(j)) << "]" << ((j==reps) ? "" : "\r");
                std::cout.flush();
            }
            results.push_back(elapsed_total/(static_cast<double>(reps)));
            std::cout << "\n" << std::endl;
        }
        std::cout << "" << std::endl;
        printAsPoints(results);
    }

    std::cout << "\nFinished." << std::endl;

    return EXIT_SUCCESS;
}
