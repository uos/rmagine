#include <iostream>

#include <rmagine/simulation/PinholeSimulatorOptix.hpp>
#include <rmagine/map/optix/optix_shapes.h>
#include <rmagine/map/OptixMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/prints.h>

#include <stdexcept>
#include <cassert>


using namespace rmagine;

OptixMapPtr make_map()
{
    OptixScenePtr scene = std::make_shared<OptixScene>();

    OptixGeometryPtr mesh = std::make_shared<OptixCube>();
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    return std::make_shared<OptixMap>(scene);
}   

int main(int argc, char** argv)
{
    // make synthetic map
    OptixMapPtr map = make_map();
    
    auto model = example_pinhole();
    PinholeSimulatorOptix sim;
    {
        sim.setMap(map);
        sim.setModel(model);
    }

    using ResultT = IntAttrAny<VRAM_CUDA>;

    ResultT result;
    resize_memory_bundle<VRAM_CUDA>(result, model.getWidth(), model.getHeight(), 100);

    Memory<Transform, RAM> T(100);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    Memory<Transform, VRAM_CUDA> T_ = T;

    std::cout << "Simulate!" << std::endl;
    
    for(size_t i=0; i<100; i++)
    {
        sim.simulate(T_, result);

        // std::cout << "bla" << std::endl;
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
            RM_THROW(OptixException, ss.str());                                                            
        }
    }

    std::cout << "Done simulating." << std::endl;

    return 0;
}