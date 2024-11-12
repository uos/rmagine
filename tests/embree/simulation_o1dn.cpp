#include <iostream>

#include <rmagine/simulation/O1DnSimulatorEmbree.hpp>
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
    
    auto model = example_o1dn();
    O1DnSimulatorEmbree sim;
    {
        sim.setMap(map);
        sim.setModel(model);
    }

    IntAttrAll<RAM> result;
    resize_memory_bundle<RAM>(result, model.getWidth(), model.getHeight(), 100);

    Memory<Transform, RAM> T(100);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    std::cout << "Simulate!" << std::endl;
    
    // Memory<float, RAM> last_scan(1);
    for(size_t i=0; i<1000; i++)
    {
        sim.simulate(T, result);

        Memory<float, RAM> last_scan = result.ranges(
            model.size() * 99,
            model.size() * 100
        );

        float range = last_scan[model.getBufferId(0,0)];
        Vector3 dir = model.getDirection(0,0);
        Vector3 orig = model.getOrigin(0,0);
        Vector3 point = orig + dir * range;

        Vector3 diff = point - Vector3{0.256732, 0.5, -0.1};

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