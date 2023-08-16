#include <iostream>

#include <rmagine/simulation/O1DnSimulatorOptix.hpp>
#include <rmagine/map/optix/optix_shapes.h>
#include <rmagine/map/OptixMap.hpp>
#include <rmagine/types/sensors.h>
#include <rmagine/util/prints.h>

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
    
    auto model = example_o1dn();
    O1DnSimulatorOptix sim;
    {
        sim.setMap(map);
        sim.setModel(model);
    }

    size_t Nposes = 100;
    size_t Nsteps = 1000;

    IntAttrAny<VRAM_CUDA> result;
    resize_memory_bundle<VRAM_CUDA>(result, model.getWidth(), model.getHeight(), 100);

    Memory<Transform, RAM> T(Nposes);
    for(size_t i=0; i<T.size(); i++)
    {
        T[i] = Transform::Identity();
    }

    Memory<Transform, VRAM_CUDA> T_ = T;

    std::cout << "Simulate to VRAM_CUDA ..." << std::endl;
    
    // Memory<float, RAM> last_scan(1);
    for(size_t i=0; i<Nsteps; i++)
    {
        sim.simulate(T_, result);

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
            RM_THROW(OptixException, ss.str());                                                          
        }

        Memory<Vector> last_normals = result.normals(
            model.size() * 99,
            model.size() * 100
        );

        float normal_error = (last_normals[0] - Vector{0.0, -1.0, 0.0}).l2norm();
        if(normal_error > 0.0001)                                              
        {                   
            std::stringstream ss;
            ss << "Simulated normal error is too high: " << error;
            RM_THROW(OptixException, ss.str());
        }
    }

    

    IntAttrAny<UNIFIED_CUDA> result2;
    resize_memory_bundle<UNIFIED_CUDA>(result2, 
        model.getWidth(), model.getHeight(), Nposes);

    std::cout << "Simulate to UNIFIED_CUDA ..." << std::endl;

    for(size_t i=0; i<Nsteps; i++)
    {
        sim.simulate(T_, result2);

        auto last_scan = result2.ranges(
            model.size() * 99,
            model.size() * 100
        );

        float error = std::fabs(last_scan[0] - 0.570887);

        if(error > 0.0001)                                  
        {                   
            std::stringstream ss;
            ss << "Simulated scan error is too high: " << error;
            RM_THROW(OptixException, ss.str());
        }
    }


    using MixedResults = Bundle<
        Ranges<UNIFIED_CUDA>,
        Normals<VRAM_CUDA>
    >;

    MixedResults result3;
    // this is a bit ugly. TODO: better?
    resize_memory_bundle<UNIFIED_CUDA>(result3, 
        model.getWidth(), model.getHeight(), Nposes);
    resize_memory_bundle<VRAM_CUDA>(result3, 
        model.getWidth(), model.getHeight(), Nposes);

    std::cout << "Simulate to mixed memory ..." << std::endl;


    for(size_t i=0; i<Nsteps; i++)
    {
        sim.simulate(T_, result3);

        auto last_scan = result3.ranges(
            model.size() * 99,
            model.size() * 100
        );

        float error = std::fabs(last_scan[0] - 0.570887);
                                       
        if(error > 0.0001)                                              
        {                   
            std::stringstream ss;
            ss << "Simulated scan error is too high: " << error;
            RM_THROW(OptixException, ss.str());
        }

        Memory<Vector> last_normals = result3.normals(
            model.size() * 99,
            model.size() * 100
        );

        float normal_error = (last_normals[0] - Vector{0.0, -1.0, 0.0}).l2norm();
        if(normal_error > 0.0001)                                              
        {                   
            std::stringstream ss;
            ss << "Simulated normal error is too high: " << error;
            RM_THROW(OptixException, ss.str());
        }
    }

    std::cout << "Done simulating." << std::endl;


    return 0;
}