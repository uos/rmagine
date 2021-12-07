#include <iostream>

// General mamcl includes
#include <imagine/types/sensor_models.h>
#include <imagine/util/StopWatch.hpp>

// CPU
#include <imagine/simulation/EmbreeSimulator.hpp>
#include <imagine/types/Memory.hpp>

// GPU
#if defined WITH_OPTIX
#include <imagine/simulation/OptixSimulator.hpp>
#include <imagine/types/MemoryCuda.hpp>
#endif

#include <iomanip>

#include <fstream>

using namespace imagine;

Memory<LiDARModel, RAM> velodyne_model()
{
    Memory<LiDARModel, RAM> model;
    model->theta.min = -M_PI;
    model->theta.max = M_PI; 
    model->theta.size = 440;
    model->theta.computeStep();
    
    model->phi.min = -0.261799;
    model->phi.max = 0.261799;
    model->phi.size = 16;
    model->phi.computeStep();
    
    model->range.min = 0.5;
    model->range.max = 130.0;
    return model;
}

// std::string getTimeString()
// {

// }

int main(int argc, char** argv)
{
    std::cout << "Imagine Evaluation" << std::endl;

    // what to do here:
    // Measure runtime of simulation
    // Sample random poses. Use the same for GPU and CPU
    // Use a given mesh (argv[0])
    // simulate a scanner model from N different poses
    // N elem of [1, 10, 100, 1000, 10000] 
    // for each N collect 10000 runtimes -> save into file

    // Parameters for evaluation:
    // - meshfile (string)
    // - Nposes (list of uint) or maybe (min, max, inc)?
    // - Nmeasurements (uint)

    // producing output of rows:
    // device_name triangles vertices poses sensor_width sensor_height runtime0 runtime1 ...

    if(argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " mesh_file device_type [device_id]" << std::endl;
    
        std::cout << std::endl;
        std::cout << "- mesh_file (string): path to mesh" << std::endl;
        std::cout << " - device_type (string): elem of cpu|gpu" << std::endl;
    }

    std::string meshfile = argv[1];

    std::string device = argv[2];
    int device_id = 0;

    if(argc > 3)
    {
        // device id specified
        device_id = atoi(argv[3]);
    }

    // check arguments

    if(device != std::string("gpu") && device != std::string("cpu") )
    {
        std::cout << "device_type: '" << device << "' unknown." << std::endl; 
        return 0;
    }


    if(device == "gpu")
    {
        #ifndef WITH_OPTIX
        // correct
        std::cout << "Cannot use device '" << device << "'. Compile it by enabling OptiX." << std::endl;
        return 0;
        #endif // WITH_OPTIX
    }

    std::cout << "Arguments: " << std::endl;
    std::cout << "- meshfile: " << meshfile << std::endl;
    std::cout << "- device: " << device << std::endl;
    std::cout << "- device_id: " << device_id << std::endl;

    


    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::stringstream ss;
    ss << "imagine_evaluation_" << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S") << ".csv";
    std::cout << "Outfile: " << ss.str() << std::endl;

    std::ofstream outfile;
    outfile.open(ss.str());

    size_t Nmeasurements = 10;
    size_t minPoses = 1000;
    size_t maxPoses = 10000;
    size_t incPoses = 1000;

    // Prepare
    Memory<Transform, RAM> Tsb;
    Tsb->R.x = 0.0;
    Tsb->R.y = 0.0;
    Tsb->R.z = 0.0;
    Tsb->R.w = 1.0;
    Tsb->t.x = 0.0;
    Tsb->t.y = 0.0;
    Tsb->t.z = 0.0;

    // Get Sensor Model
    Memory<LiDARModel, RAM> model = velodyne_model();
    
    if(device == "cpu")
    {
        EmbreeMapPtr cpu_mesh = importEmbreeMap(meshfile);
        EmbreeSimulatorPtr cpu_sim(new EmbreeSimulator(cpu_mesh));

        cpu_sim->setTsb(Tsb);
        cpu_sim->setModel(model);


        size_t Ntriangles = 0;
        size_t Nvertices = 0;

        for(size_t i=0; i<cpu_mesh->meshes.size(); i++)
        {
            Ntriangles += cpu_mesh->meshes[i].Nfaces;
            Nvertices += cpu_mesh->meshes[i].Nvertices;
        }

        for(size_t Nposes = minPoses; Nposes < maxPoses; Nposes += incPoses)
        {
            std::cout << Nposes << "/" << maxPoses << std::endl;
            size_t Nrays = Nposes * model->theta.size * model->phi.size;
            
            StopWatch sw;
            double el;

            // Define Transforms Base to Map (Poses)
            Memory<Transform, RAM> Tbm(Nposes);
            for(size_t i=0; i<Tbm.size(); i++)
            {
                // TODO: random
                Tbm[i] = Tsb[0];
            }

            using ResultT = Bundle<Ranges<RAM> >;

            ResultT res;
            res.ranges.resize(Nrays);

            // producing output of rows:
            // device_name triangles vertices poses sensor_width sensor_height runtime0 runtime1 ...



            outfile << device << "," << Ntriangles << "," << Nvertices << "," << Nposes << "," << model->theta.size << "," << model->phi.size << ",";  
            
            sw();
            cpu_sim->simulate<ResultT>(Tbm, res);
            el = sw();
            outfile << el;

            for(size_t meas_id = 0; meas_id < Nmeasurements; meas_id++)
            {
                sw();
                cpu_sim->simulate<ResultT>(Tbm, res);
                el = sw();

                outfile << "," << el;

            }
            outfile << "\n";
        }
    } else if(device == "gpu") {
        #if defined WITH_OPTIX

        cudaDeviceProp info;
        CUDA_CHECK( cudaGetDeviceProperties(&info, device_id) );



        std::cout << info.name << std::endl;


        OptixMapPtr gpu_mesh = importOptixMap(meshfile, device_id);
        OptixSimulatorPtr gpu_sim(new OptixSimulator(gpu_mesh));

        gpu_sim->setTsb(Tsb);
        gpu_sim->setModel(model);

        using ResultT = Bundle<Ranges<VRAM_CUDA> >;
        gpu_sim->preBuildProgram<ResultT>();


        size_t Ntriangles = 0;
        size_t Nvertices = 0;

        

        for(size_t i=0; i<gpu_mesh->meshes.size(); i++)
        {
            Ntriangles += gpu_mesh->meshes[i].faces.size();
            Nvertices += gpu_mesh->meshes[i].vertices.size();
        }


        for(size_t Nposes = minPoses; Nposes <= maxPoses; Nposes += incPoses)
        {
            std::cout << Nposes << "/" << maxPoses << std::endl;
            size_t Nrays = Nposes * model->theta.size * model->phi.size;
            
            StopWatch sw;
            double el;

            // Define Transforms Base to Map (Poses)
            Memory<Transform, RAM> Tbm(Nposes);
            for(size_t i=0; i<Tbm.size(); i++)
            {
                // TODO: random
                Tbm[i] = Tsb[0];
            }
            Memory<Transform, VRAM_CUDA> Tbm_gpu;
            Tbm_gpu = Tbm;

            

            ResultT res;
            res.ranges.resize(Nrays);

            // producing output of rows:
            // device_name triangles vertices poses sensor_width sensor_height runtime0 runtime1 ...

            outfile << info.name << "," << Ntriangles << "," << Nvertices << "," << Nposes << "," << model->theta.size << "," << model->phi.size << ",";  
            
            sw();
            gpu_sim->simulate<ResultT>(Tbm_gpu, res);
            el = sw();
            outfile << el;

            for(size_t meas_id = 0; meas_id < Nmeasurements; meas_id++)
            {
                sw();
                gpu_sim->simulate<ResultT>(Tbm_gpu, res);
                cudaDeviceSynchronize();
                el = sw();

                outfile << "," << el;

            }
            outfile << "\n";
        }

        #endif // WITH_OPTIX
    }
    
    outfile.close();

    // if(device == "cpu")
    // {

    // } else if(device == "gpu") {
    //     #if defined WITH_OPTIX
        
    //     using ResultT = Bundle<Ranges<VRAM_CUDA> >;



    //     #else
    //     std::cout << "Cannot use device '" << device << "'. Compile it by enabling OptiX." << std::endl;
    //     #endif // WITH_OPTIX
    // } else {
    //     std::cout << "Device '" << device << "' unknown" << std::endl;
    // }


    return 0;
}
