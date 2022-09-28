#include <iostream>

#include <rmagine/simulation/OnDnSimulatorEmbree.hpp>
#include <rmagine/map/embree/embree_shapes.h>
#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/exceptions.h>



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
    // make synthetic map
    EmbreeMapPtr map = make_map();
    
    auto model = example_ondn();
    OnDnSimulatorEmbree sim;
    {
        
        sim.setMap(map);
        sim.setModel(model);
    }

    size_t Nposes = 100;
    size_t Nsteps = 1000;


    IntAttrAny<RAM> result;
    resizeMemoryBundle<RAM>(result, model.getWidth(), model.getHeight(), Nposes);

    Memory<Transform, RAM> T(Nposes);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    std::cout << "Simulate!" << std::endl;
    
    
    for(size_t i=0; i<Nsteps; i++)
    {
        sim.simulate(T, result);

        Memory<float, RAM> last_scan = result.ranges(
            model.size() * 99,
            model.size() * 100
        );

        float error = std::fabs(last_scan[0] - 0.5);
                                                        
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