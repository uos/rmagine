#include <iostream>

#include <rmagine/simulation/PinholeSimulatorEmbree.hpp>
#include <rmagine/map/embree/embree_shapes.h>
#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/prints.h>
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
    
    auto model = example_pinhole();
    PinholeSimulatorEmbree sim;
    {
        sim.setMap(map);
        sim.setModel(model);
    }

    using ResultT = IntAttrAll<RAM>;

    ResultT result;
    resize_memory_bundle<RAM>(result, model.getWidth(), model.getHeight(), 100);

    Memory<Transform, RAM> T(100);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    std::cout << "Simulate!" << std::endl;
    
    for(size_t i=0; i<10; i++)
    {
        sim.simulate(T, result);

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
            RM_THROW(EmbreeException, ss.str());                                                        
        }
    }

    std::cout << "Done simulating." << std::endl;

    return 0;
}