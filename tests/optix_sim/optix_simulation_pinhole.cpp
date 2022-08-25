#include <iostream>

#include <rmagine/simulation/PinholeSimulatorOptix.hpp>
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


PinholeModel sensor_model()
{
    PinholeModel model;
    model.width = 400;
    model.height = 300;
    model.c[0] = 200.0;
    model.c[1] = 150.0;
    model.f[0] = 1000.0;
    model.f[1] = 1000.0;
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

int main(int argc, char** argv)
{
    // make synthetic map
    OptixMapPtr map = make_map();
    
    auto model = sensor_model();
    PinholeSimulatorOptix sim;
    {
        sim.setMap(map);
        sim.setModel(model);
    }

    using ResultT = IntAttrAny<VRAM_CUDA>;

    // using ResultT = Bundle<
    //     Ranges<VRAM_CUDA>
    // >;

    ResultT result;
    resizeMemoryBundle<VRAM_CUDA>(result, model.getWidth(), model.getHeight(), 100);

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

        float error = std::fabs(last_scan[0] - 0.515388);
                                                        
        if(error > 0.0001)                                              
        {                                                           
            std::stringstream ss;
            ss << LOC_STRING() << ": Simulated scan error is too high: " << error;  
            throw std::runtime_error( ss.str() );                                                              
        }
    }

    std::cout << "Done simulating." << std::endl;

    return 0;
}