#include <iostream>

#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/map/optix/optix_shapes.h>
#include <rmagine/map/OptixMap.hpp>

#include <stdexcept>
#include <cassert>

using namespace rmagine;


#define LOC_STRING() \
    std::string( __FILE__ )             \
    + std::string( ":" )                  \
    + std::to_string( __LINE__ )          \
    + std::string( " in " )               \
    + std::string( __PRETTY_FUNCTION__ ) 

SphericalModel sensor_model()
{
    SphericalModel model;
    model.theta.min = -M_PI;
    model.theta.inc = 0.4 * M_PI / 180.0;
    model.theta.size = 900;

    model.phi.min = -0.0 * M_PI / 180.0;
    model.phi.inc = 2.0 * M_PI / 180.0;
    model.phi.size = 16;
    
    model.range.min = 0.0;
    model.range.max = 100.0;
    return model;
}

OptixMapPtr make_map()
{
    OptixScenePtr scene = std::make_shared<OptixScene>();

    OptixGeometryPtr mesh = std::make_shared<OptixCube>();
    mesh->commit();

    scene->add(mesh);

    scene->commit();

    return std::make_shared<OptixMap>(scene);
}   

void test_basic()
{
    SphereSimulatorOptix sim;


    // make synthetic map
    OptixMapPtr map = make_map();
    sim.setMap(map);
    
    auto model = sensor_model();
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
    
    // Memory<float, RAM> last_scan(1);
    for(size_t i=0; i<1000; i++)
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
            ss << LOC_STRING() << ": Simulated scan error is too high: " << error;  
            throw std::runtime_error( ss.str() );                                                              
        }
    }

    std::cout << "Done simulating." << std::endl;
}

void test_empty_scene()
{

    

    OptixScenePtr scene = std::make_shared<OptixScene>();
    scene->commit();
    OptixMapPtr map = std::make_shared<OptixMap>(scene);

    
    // s

    auto model = sensor_model();
    
    Memory<Transform, RAM> T(100);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    Memory<Transform, VRAM_CUDA> T_ = T;


    SphereSimulatorOptix sim;
    {
        sim.setMap(map);
        sim.setModel(model);
    }
    

    IntAttrAny<VRAM_CUDA> result;
    resizeMemoryBundle<VRAM_CUDA>(result, model.getWidth(), model.getHeight(), 1);
    
    // emptry scene. the results should be invalid. example: range must be range_max + 1
    sim.simulate(T_, result);




}

int main(int argc, char** argv)
{
    test_basic();
    test_empty_scene();



    return 0;
}