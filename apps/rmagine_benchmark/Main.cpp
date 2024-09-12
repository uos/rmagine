#include <iostream>

// Core rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>


// CPU - Embree
#if defined WITH_EMBREE
#include <rmagine/simulation/SphereSimulatorEmbree.hpp>
#include <rmagine/simulation/PinholeSimulatorEmbree.hpp>
#endif

// GPU - Optix
#if defined WITH_OPTIX
#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/types/MemoryCuda.hpp>
#endif


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

int main(int argc, char** argv)
{
    std::cout << "Rmagine Benchmark";
    
    int device_id = 0;
    std::string device;

    #if defined WITH_EMBREE
    std::cout << " CPU (Embree)";
    device = "cpu";
    #elif defined WITH_OPTIX
    std::cout << " GPU (OptiX)";
    device = "gpu";
    #else
    Either Embree or OptiX must be defined // compile time error
    #endif
    
    std::cout << std::endl;

    // Total runtime of the Benchmark in seconds
    double benchmark_duration = 10.0;
    // Poses to check per call
    size_t Nposes = 10 * 1024;

    // minimum 2 arguments
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " mesh_file" << std::endl;
        return 0;
    }

    std::string path_to_mesh = argv[1];
    // std::string device = argv[2];
    
    std::cout << "Inputs: " << std::endl;
    std::cout << "- mesh: " << path_to_mesh << std::endl;

    StopWatch sw;
    double elapsed;
    double elapsed_total = 0.0;

    Memory<LiDARModel, RAM> model = velodyne_model();

    std::cout << "Unit: 1 Velodyne scan (velo) = " << model[0].size() << " Rays" << std::endl;

    if(device == "cpu")
    {
        #if defined WITH_EMBREE
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


        // Load mesh
        EmbreeMapPtr cpu_mesh = import_embree_map(path_to_mesh);
        
        SphereSimulatorEmbreePtr cpu_sim(new SphereSimulatorEmbree(cpu_mesh));

        cpu_sim->setTsb(Tsb);
        cpu_sim->setModel(model);

        // Define what to simulate
        double velos_per_second_mean = 0.0;

        // std::cout << "- range of last ray: " << cpu_sim->simulateRanges(Tbm)[Tbm.size() * model->phi.size * model->theta.size - 1] << std::endl;
        std::cout << "-- Starting Benchmark --" << std::endl;

        // predefine result buffer
        // Memory<float, RAM> res(Tbm.size() * model->phi.size * model->theta.size);

        using ResultT = Bundle<
            Ranges<RAM>
        >;

        ResultT res;
        res.ranges.resize(Tbm.size() * model->phi.size * model->theta.size);

        int run = 0;
        while(elapsed_total < benchmark_duration)
        {
            double n_dbl = static_cast<double>(run) + 1.0;
            // Simulate
            sw();
            cpu_sim->simulate(Tbm, res);
            elapsed = sw();
            elapsed_total += elapsed;
            double velos_per_second = static_cast<double>(Nposes) / elapsed;
            velos_per_second_mean = (n_dbl - 1.0)/(n_dbl) * velos_per_second_mean + (1.0 / n_dbl) * velos_per_second; 
            

            std::cout 
            << std::fixed
            << "[ " << int((elapsed_total / benchmark_duration)*100.0) << "%" << " - " 
            << velos_per_second << " velos/s" 
            << ", mean: " << velos_per_second_mean << " velos/s] \r";
            std::cout.flush();

            run++;
        }

        std::cout << std::endl;
        std::cout << "Result: " << velos_per_second_mean << " velos/s" << std::endl;
        

        // clean up
        #else // WITH_EMBREE

        std::cout << "cpu benchmark not possible. Compile with Embree support." << std::endl;

        #endif

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
        
        // OptixMapPtr gpu_mesh = import_optix_map(path_to_mesh);
        
        // // Load mesh
        // AssimpIO io;
        // const aiScene* ascene = io.ReadFile(path_to_mesh, 0);

        // if(!ascene)
        // {
        //     std::cerr << io.Importer::GetErrorString() << std::endl;
        // }

        // OptixScenePtr scene = make_optix_scene(ascene);
        // scene->commit();

        // std::cout << "Top Level geometries: " << scene->geometries().size() << std::endl;

        OptixMapPtr gpu_mesh = import_optix_map(path_to_mesh);
        SphereSimulatorOptixPtr gpu_sim = std::make_shared<SphereSimulatorOptix>(gpu_mesh);

        gpu_sim->setTsb(Tsb);
        gpu_sim->setModel(model);

        // upload
        Memory<Transform, VRAM_CUDA> Tbm_gpu;
        Tbm_gpu = Tbm;

        // Define what to simulate

        Memory<float, RAM> ranges_cpu;
        Memory<unsigned int, RAM> geom_ids_cpu;
        Memory<unsigned int, RAM> obj_ids_cpu;

        using ResultT = Bundle<
            Ranges<VRAM_CUDA>
        >;
        // using ResultT = IntAttrAll<VRAM_CUDA>;
        // using ResultT = Bundle<
        //     Ranges<VRAM_CUDA>
        //     ,Normals<VRAM_CUDA>
        //     // ,GeomIds<VRAM_CUDA>
        //     // ,ObjectIds<VRAM_CUDA>
        // >;

        ResultT res;

        res.ranges.resize(Tbm.size() * model->size());
        // resize_memory_bundle<VRAM_CUDA>(res, Tbm.size(), model->phi.size, model->theta.size);
        
        gpu_sim->simulate(Tbm_gpu, res);
        ranges_cpu = res.ranges;
        // geom_ids_cpu = res.geom_ids;
        // obj_ids_cpu = res.object_ids;
        
        std::cout << "Last Ray:" << std::endl;
        
        std::cout << "- range: " << ranges_cpu[Tbm.size() * model->size() - 1] << std::endl;
        // std::cout << "- geom id: " << geom_ids_cpu[Tbm.size() * model->size() - 1] << std::endl;
        // std::cout << "- obj id: " << obj_ids_cpu[Tbm.size() * model->size() - 1] << std::endl;
        
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
            // scene->commit();
            gpu_sim->simulate(Tbm_gpu, res);
            elapsed = sw();
            elapsed_total += elapsed;
            double velos_per_second = static_cast<double>(Nposes) / elapsed;
            velos_per_second_mean = (n_dbl - 1.0)/(n_dbl) * velos_per_second_mean + (1.0 / n_dbl) * velos_per_second; 
            
            // std::cout
            // << "[ " << int((elapsed_total / benchmark_duration)*100.0) << "%" << " - " 
            // << "velos/s: " << velos_per_second 
            // << ", mean: " << velos_per_second_mean 
            // << "] \r";
            // std::cout.flush();

            std::cout 
            << std::fixed
            << "[ " << int((elapsed_total / benchmark_duration)*100.0) << "%" << " - " 
            << velos_per_second << " velos/s" 
            << ", mean: " << velos_per_second_mean << " velos/s] \r";
            std::cout.flush();

            run++;
        }

        std::cout << std::endl;
        std::cout << "Result: " << velos_per_second_mean << " velos/s" << std::endl;
        #else // WITH_OPTIX
            std::cout << "gpu benchmark not possible. Compile with OptiX support." << std::endl;

        #endif
    } else {
        std::cout << "Device " << device << " unknown" << std::endl;
    }

    return 0;
}
