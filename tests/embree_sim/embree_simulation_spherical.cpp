#include <iostream>

#include <rmagine/simulation/SphereSimulatorEmbree.hpp>
#include <rmagine/map/embree/embree_shapes.h>
#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>


using namespace rmagine;


EmbreeMapPtr make_map()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();

    EmbreeGeometryPtr mesh = std::make_shared<EmbreeCube>();
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    return std::make_shared<EmbreeMap>(scene);
}

int main(int argc, char** argv)
{
    SphereSimulatorEmbree sim;

    

    // make synthetic map
    EmbreeMapPtr map = make_map();
    sim.setMap(map);
    
    auto model = example_spherical();
    sim.setModel(model);

    IntAttrAll<RAM> result;
    resize_memory_bundle<RAM>(result, model.getWidth(), model.getHeight(), 100);

    Memory<Transform, RAM> T(100);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    std::cout << "Simulate!" << std::endl;
    
    for(size_t i=0; i<20; i++)
    {
        sim.simulate(T, result);

        Memory<float, RAM> last_scan = result.ranges(
            model.size() * 99,
            model.size() * 100
        );

        float range = last_scan[model.getBufferId((model.phi.size) / 2, 0)];
        // std::cout << range << std::endl;
        float error = std::fabs(range - 0.500076);
                                                        
        if(error > 0.0001)                                              
        {                                                           
            std::stringstream ss;
            ss << "Simulated scan error is too high: " << error;
            RM_THROW(EmbreeException, ss.str());
        }
    }

    std::cout << "Done simulating." << std::endl;

    return 0;
}