#include <iostream>

#include <rmagine/simulation/PinholeSimulatorVulkan.hpp>
#include <rmagine/map/vulkan/vulkan_shapes.hpp>
#include <rmagine/map/VulkanMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/exceptions.h>

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
    
    auto model = example_pinhole();
    PinholeSimulatorVulkan sim;
    {
        sim.setMap(map);
        sim.setModel(model);
    }

    using ResultT = IntAttrAll<DEVICE_LOCAL_VULKAN>;

    ResultT result;
    resize_memory_bundle<DEVICE_LOCAL_VULKAN>(result, model.getWidth(), model.getHeight(), 100);

    Memory<Transform, RAM> T(100);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    Memory<Transform, DEVICE_LOCAL_VULKAN> T_ = T;

    std::cout << "Simulate!" << std::endl;
    
    for(size_t i=0; i<100; i++)
    {
        sim.simulate(T_, result);

        Memory<float, RAM> last_scan = result.ranges(
            model.size() * 99,
            model.size() * 100
        );

        float range = last_scan[model.getBufferId(0,0)];
        Vector dir = model.getDirection(0,0);
        Vector3 point = dir * range;
        Vector3 diff = point - Vector3{0.5, 0.5, 0.375};

        float error = std::fabs(diff.x) + std::fabs(diff.y) + std::fabs(diff.z);
                                                        
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