#include <iostream>

#include <rmagine/simulation/OnDnSimulatorOptix.hpp>
#include <rmagine/map/optix/optix_shapes.h>
#include <rmagine/map/OptixMap.hpp>
#include <rmagine/types/sensors.h>

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
    
    auto model = example_ondn();
    OnDnSimulatorOptix sim;
    {
        
        sim.setMap(map);
        sim.setModel(model);
    }

    size_t Nposes = 100;
    size_t Nsteps = 1000;


    IntAttrAny<VRAM_CUDA> result;
    resize_memory_bundle<VRAM_CUDA>(result, model.getWidth(), model.getHeight(), Nposes);

    Memory<Transform, RAM> T(Nposes);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    Memory<Transform, VRAM_CUDA> T_ = T;

    std::cout << "Simulate!" << std::endl;
    
    
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
            RM_THROW(OptixException, ss.str());
        }
    }

    std::cout << "Done simulating." << std::endl;

    return 0;
}