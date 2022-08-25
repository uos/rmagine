#include <iostream>

#include <rmagine/simulation/O1DnSimulatorOptix.hpp>
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

Memory<SphericalModel, RAM> example_spherical_model()
{
    Memory<LiDARModel, RAM> model(1);
    model->theta.min = -M_PI;
    model->theta.inc = 0.4 * M_PI / 180.0;
    model->theta.size = 900;

    model->phi.min = -15.0 * M_PI / 180.0;
    model->phi.inc = 2.0 * M_PI / 180.0;
    model->phi.size = 16;
    
    model->range.min = 0.0;
    model->range.max = 130.0;
    return model;
}


O1DnModel sensor_model()
{
    // represent spherical model as custom model to compare results
    // build model out of two velo models
    auto velo_model = example_spherical_model();

    O1DnModel model;
        
    size_t W = velo_model->getWidth();
    size_t H = velo_model->getHeight() * 2;

    model.width = W;
    model.height = H;
    model.range = velo_model->range;

    model.orig.x = 0.0;
    model.orig.y = 0.0;
    model.orig.z = 0.5;
    model.dirs.resize(W * H);

    for(size_t vid=0; vid<velo_model->getHeight(); vid++)
    {
        for(size_t hid=0; hid<velo_model->getWidth(); hid++)
        {
            const Vector ray = velo_model->getDirection(vid, hid);
            unsigned int loc_id_1 = model.getBufferId(vid, hid);
            model.dirs[loc_id_1] = ray;

            const Vector ray_flipped = {ray.x, ray.z, ray.y};
            unsigned int loc_id_2 = model.getBufferId(vid + velo_model->getHeight(), hid);
            model.dirs[loc_id_2] = ray_flipped;
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

int main(int argc, char** argv)
{
    // make synthetic map
    OptixMapPtr map = make_map();
    
    auto model = sensor_model();
    O1DnSimulatorOptix sim;
    {
        sim.setMap(map);
        sim.setModel(model);
    }

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

        float error = std::fabs(last_scan[0] - 0.517638);                     
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