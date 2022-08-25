#include <iostream>

#include <rmagine/simulation/OnDnSimulatorOptix.hpp>
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

OnDnModel sensor_model()
{
    OnDnModel model;

    model.height = 10;
    model.width = 100;

    model.range.min = 0.0;
    model.range.max = 100.0;
    
    model.origs.resize(model.width * model.height);
    model.dirs.resize(model.width * model.height);

    

    for(size_t vid=0; vid<model.getHeight(); vid++)
    {
        Vector orig = {0.0, 0.0, 0.0};
        // v equally distributed between -0.5 and 0.5
        float v = static_cast<float>(vid) / 100.f;
        orig.z = v;
        for(size_t hid=0; hid<model.getWidth(); hid++)
        {
            // h from 0 to 2PI
            float h = static_cast<float>(hid) / 100.f;
            orig.y = h;
            Vector ray = {1.0, 0.0, 0.0};
            unsigned int loc_id = model.getBufferId(vid, hid);
            model.origs[loc_id] = orig;
            model.dirs[loc_id] = ray;
        }
    }

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

void test1()
{
    // make synthetic map
    OptixMapPtr map = make_map();
    
    auto model = sensor_model();
    OnDnSimulatorOptix sim;
    {
        
        sim.setMap(map);
        sim.setModel(model);
    }

    size_t Nposes = 100;
    size_t Nsteps = 1000;


    IntAttrAny<VRAM_CUDA> result;
    resizeMemoryBundle<VRAM_CUDA>(result, model.getWidth(), model.getHeight(), Nposes);

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
            ss << LOC_STRING() << ": Simulated scan error is too high: " << error;  
            throw std::runtime_error( ss.str() );                                                              
        }
    }

    std::cout << "Done simulating." << std::endl;
}

void test2()
{
    OptixScenePtr scene = std::make_shared<OptixScene>();
    scene->commit();
    OptixMapPtr map = std::make_shared<OptixMap>(scene);


    // s

    // auto model = sensor_model();
    
    // IntAttrAny<VRAM_CUDA> result;
    // resizeMemoryBundle<VRAM_CUDA>(result, model.getWidth(), model.getHeight(), Nposes);
    
}

int main(int argc, char** argv)
{
    test1();
    test2();

    return 0;
}