#include <iostream>

// General mamcl includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>

// CPU
#include <rmagine/simulation/SphereSimulatorEmbree.hpp>
#include <rmagine/simulation/PinholeSimulatorEmbree.hpp>

#include <rmagine/types/Memory.hpp>
#include <rmagine/map/embree/embree_shapes.h>

// GPU
#if defined WITH_OPTIX
#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/types/MemoryCuda.hpp>
#endif

#include <iomanip>

using namespace rmagine;

Memory<LiDARModel, RAM> velodyne_model()
{
    Memory<LiDARModel, RAM> model(1);
    model->theta.min = -M_PI;
    model->theta.inc = 0.4 * M_PI / 180.0;
    model->theta.size = 900;

    model->phi.min = -15.0 * M_PI / 180.0;
    model->phi.inc = 2.0 * M_PI / 180.0;
    model->phi.size = 16;
    
    model->range.min = 0.1;
    model->range.max = 130.0;
    return model;
}


void printRaycast(EmbreeScenePtr scene, Vector3 orig, Vector3 dir)
{
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    // std::cout << "[printRaycast()] called." << std::endl;
    // std::cout << "- instance stack size: " << context.instStackSize << std::endl;

    RTCRayHit rayhit;
    rayhit.ray.org_x = orig.x;
    rayhit.ray.org_y = orig.y;
    rayhit.ray.org_z = orig.z;
    rayhit.ray.dir_x = dir.x;
    rayhit.ray.dir_y = dir.y;
    rayhit.ray.dir_z = dir.z;
    rayhit.ray.tnear = 0;
    rayhit.ray.tfar = INFINITY;
    rayhit.ray.mask = 0;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene->handle(), &context, &rayhit);

    std::cout << "Raycast:" << std::endl;

    std::cout << "- range: " << rayhit.ray.tfar << std::endl;

    if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
    {
        
        std::cout << "- geomID: " << rayhit.hit.geomID << std::endl;
    }

    if(rayhit.hit.instID[0] != RTC_INVALID_GEOMETRY_ID)
    {
        std::cout << "- instID: " << rayhit.hit.instID[0] << std::endl;
    }
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Benchmark" << std::endl;

    // Total runtime of the Benchmark in seconds
    double benchmark_duration = 10.0;
    // Poses to check per call
    size_t Nposes = 10 * 1024;

    // minimum 2 arguments
    if(argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " mesh_file device [device_id]" << std::endl;

        return 0;
    }

    std::string path_to_mesh = argv[1];
    std::string device = argv[2];
    int device_id = 0;
    
    if(argc > 3)
    {
        // device id specified
        device_id = atoi(argv[3]);
    }

    std::cout << "Inputs: " << std::endl;
    std::cout << "- mesh: " << path_to_mesh << std::endl;
    std::cout << "- device: " << device << std::endl;
    std::cout << "- device_id: " << device_id << std::endl; 

    StopWatch sw;
    double elapsed;
    double elapsed_total;

    if(device == "cpu")
    {
        // Define one Transform Sensor to Base
        Memory<Transform, RAM> Tsb(1);
        Tsb->R.x = 0.0;
        Tsb->R.y = 0.0;
        Tsb->R.z = 0.0;
        Tsb->R.w = 1.0;
        Tsb->t.x = 0.0;
        Tsb->t.y = 0.0;
        Tsb->t.z = 0.0;


        // Define Transforms Base to Map (Poses)
        Memory<Transform, RAM> Tbm(Nposes);
        for(size_t i=0; i<Tbm.size(); i++)
        {
            Tbm[i] = Tsb[0];
        }

        // Get Sensor Model
        Memory<LiDARModel, RAM> model = velodyne_model();
        std::cout << "theta" << std::endl;
        std::cout << "  min: " << model->theta.min << std::endl;
        std::cout << "  max: " << model->theta.max() << std::endl;
        std::cout << "  N: " << model->theta.size << std::endl;
        std::cout << "  inc: " << model->theta.inc << std::endl;
        std::cout << "phi" << std::endl;
        std::cout << "  min: " << model->phi.min << std::endl;
        std::cout << "  max: " << model->phi.max() << std::endl;
        std::cout << "  N: " << model->phi.size << std::endl;
        std::cout << "  inc: " << model->phi.inc << std::endl;

        // Load mesh
        EmbreeMapPtr cpu_mesh = importEmbreeMap(path_to_mesh);

        // // cpu_mesh->scene.reset();

        // // cpu_mesh->scene = std::make_shared<EmbreeScene>();

        // auto scene = std::make_shared<EmbreeScene>();

        // auto sphere = std::make_shared<EmbreeSphere>(1.0);
        // sphere->commit();

        // auto sphere_scene = std::make_shared<EmbreeScene>();
        // sphere_scene->add(sphere);
        // sphere_scene->commit();

        // auto sphere_inst = std::make_shared<EmbreeInstance>();
        // sphere_inst->set(sphere_scene);
        // Transform T;
        // T.setIdentity();
        // sphere_inst->setTransform(T);
        // sphere_inst->setScale({5.0, 5.0, 5.0});
        // sphere_inst->apply();
        // sphere_inst->commit();

        // scene->add(sphere_inst);
        
        // scene->commit();
        // // cpu_mesh->scene = scene;

        // std::cout << "Raycast.." << std::endl;
        // printRaycast(scene, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0});

        
        // // EmbreeMapPtr cpu_mesh;
        // std::cout << "Default constructor" << std::endl;
        // EmbreeMapPtr cpu_mesh = std::make_shared<EmbreeMap>();
        // cpu_mesh->scene = scene;

        // return 0;

        
        // return 0;
        // std::cout << "Mesh loaded to CPU." << std::endl;
        // SphereSimulatorEmbreePtr cpu_sim;
        SphereSimulatorEmbreePtr cpu_sim(new SphereSimulatorEmbree(cpu_mesh));
        // std::cout << "Initialized CPU simulator." << std::endl;

        cpu_sim->setTsb(Tsb);
        cpu_sim->setModel(model);

        // Define what to simulate

        double velos_per_second_mean = 0.0;

        std::cout << "- range of last ray: " << cpu_sim->simulateRanges(Tbm)[Tbm.size() * model->phi.size * model->theta.size - 1] << std::endl;
        std::cout << "-- Starting Benchmark --" << std::endl;

        // predefine result buffer
        Memory<float, RAM> res(Tbm.size() * model->phi.size * model->theta.size);

        int run = 0;
        while(elapsed_total < benchmark_duration)
        {
            double n_dbl = static_cast<double>(run) + 1.0;
            // Simulate
            sw();
            cpu_sim->simulateRanges(Tbm, res);
            elapsed = sw();
            elapsed_total += elapsed;
            double velos_per_second = static_cast<double>(Nposes) / elapsed;
            velos_per_second_mean = (n_dbl - 1.0)/(n_dbl) * velos_per_second_mean + (1.0 / n_dbl) * velos_per_second; 
            
            std::cout
            << "[ " << int((elapsed_total / benchmark_duration)*100.0) << "%" << " - " 
            << "velos/s: " << velos_per_second 
            << ", mean: " << velos_per_second_mean  << "] \r";
            std::cout.flush();

            run++;
        }
        std::cout << std::endl;
        std::cout << "Result: " << velos_per_second_mean << " velos/s" << std::endl;

        // clean up

    } else if(device == "gpu") {
        #if defined WITH_OPTIX

        // Define one Transform Sensor to Base
        Memory<Transform, RAM> Tsb(1);
        Tsb->R.x = 0.0;
        Tsb->R.y = 0.0;
        Tsb->R.z = 0.0;
        Tsb->R.w = 1.0;
        Tsb->t.x = 0.0;
        Tsb->t.y = 0.0;
        Tsb->t.z = 0.0;

        // Define Transforms Base to Map (Poses)
        Memory<Transform, RAM_CUDA> Tbm(Nposes);
        for(size_t i=0; i<Tbm.size(); i++)
        {
            Tbm[i] = Tsb[0];
        }

        // Get Sensor Model
        Memory<LiDARModel, RAM> model = velodyne_model();
        
        // Load mesh
        OptixMapPtr gpu_mesh = importOptixMap(path_to_mesh, device_id);
        SphereSimulatorOptixPtr gpu_sim(new SphereSimulatorOptix(gpu_mesh));

        gpu_sim->setTsb(Tsb);
        gpu_sim->setModel(model);

        // upload
        Memory<Transform, VRAM_CUDA> Tbm_gpu;
        Tbm_gpu = Tbm;

        // Define what to simulate

        Memory<float, RAM> ranges_cpu;
        using ResultT = Bundle<Ranges<VRAM_CUDA> >;
        ResultT res;
        res.ranges.resize(Tbm.size() * model->phi.size * model->theta.size);
        gpu_sim->simulate(Tbm_gpu, res);
        ranges_cpu = res.ranges;
        
        std::cout << "- range of last ray: " << ranges_cpu[Tbm.size() * model->phi.size * model->theta.size - 1] << std::endl;
        std::cout << "-- Starting Benchmark --" << std::endl;

        double velos_per_second_mean = 0.0;
        
        // predefine result buffer
        // Memory<float, VRAM_CUDA> res(Tbm.size() * model->phi.size * model->theta.size);

        int run = 0;
        while(elapsed_total < benchmark_duration)
        {
            double n_dbl = static_cast<double>(run) + 1.0;
            // Simulate
            sw();
            gpu_sim->simulate(Tbm_gpu, res);
            elapsed = sw();
            elapsed_total += elapsed;
            double velos_per_second = static_cast<double>(Nposes) / elapsed;
            velos_per_second_mean = (n_dbl - 1.0)/(n_dbl) * velos_per_second_mean + (1.0 / n_dbl) * velos_per_second; 
            
            std::cout
            << "[ " << int((elapsed_total / benchmark_duration)*100.0) << "%" << " - " 
            << "velos/s: " << velos_per_second 
            << ", mean: " << velos_per_second_mean 
            << ", rays/s: " << velos_per_second_mean * model->phi.size * model->theta.size 
            << "] \r";
            std::cout.flush();

            run++;
        }

        std::cout << std::endl;
        std::cout << "Result: " << velos_per_second_mean << " velos/s" << std::endl;
        #endif
    } else {
        std::cout << "Device " << device << " unknown" << std::endl;
    }

    return 0;
}
