#include <cmath>
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
#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/map/OptixMap.hpp>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/simulation/SphereSimulatorEmbree.hpp>
#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/types/Memory.hpp>


using namespace rmagine;



const float diff_tolerance = 0.0001;
const float max_range = 130.0;



inline float calc_diff(uint8_t& data1, uint8_t& data2)
{
    return (float)(data1 != data2);
}

inline float calc_diff(float& data1, float& data2)
{
    return std::fabs(data1 - data2);
}

inline float calc_diff(Vector3& data1, Vector3& data2)
{
    return (data1 - data2).l2norm();
}



inline bool has_hit(uint8_t& data)
{
    return data == 1;
}

inline bool has_hit(float& data)
{
    return data != (max_range + 1.0);
}

inline bool has_hit(Vector3& data)
{
    return !std::isnan(data.x);
}



inline void print(uint8_t& data)
{
    std::cout << data << std::endl;
}

inline void print(float& data)
{
    std::cout << data << std::endl;
}

inline void print(Vector3& data)
{
    std::cout << "(" << data.x << "," << data.y << "," << data.z << ")"<< std::endl;
}



template<typename DataT>
void calc_diffs(Memory<DataT, RAM>& data1, Memory<DataT, RAM>& data2, std::string data1_info, std::string data2_info)
{
    if(data1.size() != data2.size())
    {
        throw std::invalid_argument(data1_info + " - " + data2_info + " : sizes must be equal!");
    }

    uint64_t both_hits = 0;
    uint64_t data1_hits = 0;
    uint64_t data2_hits = 0;
    uint64_t neither_hits = 0;

    uint64_t both_hit_high_diff = 0;

    float max_diff = 0.0;
    float diff_avg = 0.0;
    for(size_t i = 0; i < data1.size(); i++)
    {
        bool data1_has_hit = has_hit(data1[i]);
        bool data2_has_hit = has_hit(data2[i]);

        if(data1_has_hit && data2_has_hit)
        {
            both_hits++;

            float diff = calc_diff(data1[i], data2[i]);
            if(diff > diff_tolerance)
            {
                both_hit_high_diff++;
            }
            if(diff > max_diff)
            {
                max_diff = diff;
            }

            diff_avg += diff;
        }
        else if(data1_has_hit && !data2_has_hit)
        {
            data1_hits++;
        }
        else if(!data1_has_hit && data2_has_hit)
        {
            data2_hits++;
        }
        else 
        {
            neither_hits++;
        }
    }

    diff_avg = diff_avg / ((float)both_hits);

    std::cout << data1_info << " - " << data2_info << " :" <<std::endl;

    std::cout << " - both hit: " << both_hits << std::endl;
    std::cout << " - only " << data1_info << " hits: " << data1_hits << std::endl;
    std::cout << " - only " << data2_info << " hits: " << data2_hits << std::endl;
    std::cout << " - neither hit: " << neither_hits << std::endl;

    std::cout << " - both hit with high diff (diff > " << diff_tolerance << "): " << both_hit_high_diff << std::endl;
    std::cout << " - maximum diff: " << max_diff << std::endl;
    std::cout << " - average diff (where both hit): " << diff_avg << std::endl;
    std::cout << std::endl;
}



std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<float> dist(0, 1);

Transform randomTransform()
{
    Transform tf = Transform::Identity();
    tf.R = {dist(e2), dist(e2), dist(e2), dist(e2)};
    tf.R.normalizeInplace();
    tf.t = {dist(e2), dist(e2), dist(e2)};
    tf.t.normalizeInplace();
    tf.t = tf.t * dist(e2) * 0.8;
    return tf;
}

void fillWithRandomTfs(Memory<Transform, RAM>& tbm_ram)
{
    for(size_t i = 0; i < tbm_ram.size(); i++)
    {
        tbm_ram[i] = randomTransform();
    }
}



size_t nPoses = 5000;

int main(int argc, char** argv)
{
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
        tbm_ram[i] = randomTransform();
    }
    Memory<Transform, VRAM_CUDA> tbm_vramCuda(tbm_ram.size());
    tbm_vramCuda = tbm_ram;
    Memory<Transform, DEVICE_LOCAL_VULKAN> tbm_vulkan(tbm_ram.size());
    tbm_vulkan = tbm_ram;



    // Load mesh
    EmbreeMapPtr map_embree = import_embree_map(path_to_mesh);
    OptixMapPtr map_optix = import_optix_map(path_to_mesh);
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
    model_ram->range.max = max_range;



    // Results
    using ResultEmbreeT = Bundle<
        Hits<RAM>,
        Ranges<RAM>,
        Points<RAM>,
        Normals<RAM>
    >;
    ResultEmbreeT res_ram;
    resize_memory_bundle<RAM>(res_ram, model_ram[0].getWidth(), model_ram[0].getHeight(), tbm_ram.size());

    using ResultOptixT = Bundle<
        Hits<VRAM_CUDA>,
        Ranges<VRAM_CUDA>,
        Points<VRAM_CUDA>,
        Normals<VRAM_CUDA>
    >;
    ResultOptixT res_vramCuda;
    resize_memory_bundle<VRAM_CUDA>(res_vramCuda, model_ram[0].getWidth(), model_ram[0].getHeight(), tbm_ram.size());

    using ResultVulkanT = Bundle<
        Hits<DEVICE_LOCAL_VULKAN>,
        Ranges<DEVICE_LOCAL_VULKAN>,
        Points<DEVICE_LOCAL_VULKAN>,
        Normals<DEVICE_LOCAL_VULKAN>
    >;
    ResultVulkanT res_vulkan;
    resize_memory_bundle<DEVICE_LOCAL_VULKAN>(res_vulkan, model_ram[0].getWidth(), model_ram[0].getHeight(), tbm_ram.size());



    // Simulators
    SphereSimulatorEmbreePtr sim_embree = std::make_shared<SphereSimulatorEmbree>(map_embree);
    sim_embree->setModel(model_ram);
    sim_embree->setTsb(tsb_ram);
    SphereSimulatorOptixPtr sim_optix = std::make_shared<SphereSimulatorOptix>(map_optix);
    sim_optix->setModel(model_ram);
    sim_optix->setTsb(tsb_ram);
    SphereSimulatorVulkanPtr sim_vulkan = std::make_shared<SphereSimulatorVulkan>(map_vulkan);
    sim_vulkan->setModel(model_ram);
    sim_vulkan->setTsb(tsb_ram);



    // Simulate
    sim_embree->simulate(tbm_ram, res_ram);
    sim_optix->simulate(tbm_vramCuda, res_vramCuda);
    sim_vulkan->simulate(tbm_vulkan, res_vulkan);



    // Evaluate results
    std::cout << "\nNumber of rays: " << model_ram[0].size() * nPoses << "\n" << std::endl;

    Memory<uint8_t, RAM> hits_embree = res_ram.hits;
    Memory<uint8_t, RAM> hits_optix = res_vramCuda.hits;
    Memory<uint8_t, RAM> hits_vulkan = res_vulkan.hits;
    std::cout << "\nHits:\n" << std::endl;
    calc_diffs(hits_embree, hits_vulkan, "embree", "vulkan");
    calc_diffs(hits_optix, hits_vulkan, "optix", "vulkan");
    calc_diffs(hits_embree, hits_optix, "embree", "optix");

    Memory<float, RAM> ranges_embree = res_ram.ranges;
    Memory<float, RAM> ranges_optix = res_vramCuda.ranges;
    Memory<float, RAM> ranges_vulkan = res_vulkan.ranges;
    std::cout << "\nRanges:\n" << std::endl;
    calc_diffs(ranges_embree, ranges_vulkan, "embree", "vulkan");
    calc_diffs(ranges_optix, ranges_vulkan, "optix", "vulkan");
    calc_diffs(ranges_embree, ranges_optix, "embree", "optix");

    Memory<Vector3, RAM> points_embree = res_ram.points;
    Memory<Vector3, RAM> points_optix = res_vramCuda.points;
    Memory<Vector3, RAM> points_vulkan = res_vulkan.points;
    std::cout << "\nPoints:\n" << std::endl;
    calc_diffs(points_embree, points_vulkan, "embree", "vulkan");
    calc_diffs(points_optix, points_vulkan, "optix", "vulkan");
    calc_diffs(points_embree, points_optix, "embree", "optix");

    Memory<Vector3, RAM> normals_embree = res_ram.normals;
    Memory<Vector3, RAM> normals_optix = res_vramCuda.normals;
    Memory<Vector3, RAM> normals_vulkan = res_vulkan.normals;
    std::cout << "\nNormals:\n" << std::endl;
    calc_diffs(normals_embree, normals_vulkan, "embree", "vulkan");
    calc_diffs(normals_optix, normals_vulkan, "optix", "vulkan");
    calc_diffs(normals_embree, normals_optix, "embree", "optix");



    std::cout << "\nFinished." << std::endl;
    return EXIT_SUCCESS;
}
