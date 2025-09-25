#include <iostream>

#include <rmagine/simulation/OnDnSimulatorVulkan.hpp>
#include <rmagine/map/vulkan/vulkan_shapes.hpp>
#include <rmagine/map/VulkanMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <stdexcept>
#include <cassert>

using namespace rmagine;

VulkanMapPtr make_map()
{
    VulkanScenePtr scene = std::make_shared<VulkanScene>();

    VulkanGeometryPtr mesh = std::make_shared<VulkanCube>();
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    return std::make_shared<VulkanMap>(scene);
}   

int main(int argc, char** argv)
{
    // make synthetic map
    VulkanMapPtr map = make_map();
    
    auto model = example_ondn();
    OnDnSimulatorVulkan sim;
    {
        
        sim.setMap(map);
        sim.setModel(model);
    }

    size_t Nposes = 100;
    size_t Nsteps = 1000;

    IntAttrAll<DEVICE_LOCAL_VULKAN> result;
    resize_memory_bundle<DEVICE_LOCAL_VULKAN>(result, model.getWidth(), model.getHeight(), Nposes);

    Memory<Transform, RAM> T(Nposes);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    Memory<Transform, DEVICE_LOCAL_VULKAN> T_ = T;

    std::cout << "Simulate to DEVICE_LOCAL_VULKAN ..." << std::endl;
    
    for(size_t i=0; i<Nsteps; i++)
    {
        sim.simulate(T_, result);

        Memory<float, RAM> last_scan = result.ranges(
            model.size() * 99,
            model.size() * 100
        );

        float error = std::fabs(last_scan[0] - 0.5);
                                                        
        if(error > 0.0001)                                              
        {                   
            std::stringstream ss;
            ss << "Simulated scan error is too high: " << error;
            RM_THROW(VulkanException, ss.str());
        }
    }

    std::cout << "Done simulating." << std::endl;

    return 0;
}