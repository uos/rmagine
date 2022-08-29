#include <iostream>

#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/map/optix/optix_shapes.h>
#include <rmagine/map/OptixMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/exceptions.h>


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
    SphereSimulatorOptix sim;

    // make synthetic map
    OptixMapPtr map = make_map();
    sim.setMap(map);
    
    auto model = example_spherical();
    sim.setModel(model);

    IntAttrAny<VRAM_CUDA> result;
    resizeMemoryBundle<VRAM_CUDA>(result, model.getWidth(), model.getHeight(), 100);

    Memory<Transform, RAM> T(100);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    Memory<Transform, VRAM_CUDA> T_ = T;

    std::cout << "Simulate!" << std::endl;

    // pretty_throw<Exception>("bla");
    std::string bla = "Hello";
    RM_THROW(OptixException, bla);
    
    for(size_t i=0; i<100; i++)
    {
        sim.simulate(T_, result);

        Memory<float, RAM> last_scan = result.ranges(
            model.size() * 99,
            model.size() * 100
        );

        float range = last_scan[model.getBufferId((model.phi.size) / 2, 0)];
        float error = std::fabs(range - 0.500076);
                                                        
        if(error > 0.0001)                                              
        {                             
            std::stringstream ss;
            ss << "Simulated scan error is too high: " << error;            
            // pretty_throw<OptixException>(ss.str());  
        }
    }

    std::cout << "Done simulating." << std::endl;

    return 0;
}