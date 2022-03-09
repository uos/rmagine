#include <iostream>

#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/util/StopWatch.hpp>
#include <fstream>

using namespace rmagine;

struct MyMesh
{
    unsigned int id;
};

struct MyInstance
{
    unsigned int id;
};

Memory<LiDARModel, RAM> velodyne_model()
{
    Memory<LiDARModel, RAM> model(1);
    model->theta.min = -M_PI;
    model->theta.inc = 0.4 * M_PI / 180.0;
    model->theta.size = 900;

    model->phi.min = -15.0 * M_PI / 180.0;
    model->phi.inc = 2.0 * M_PI / 180.0;
    model->phi.size = 16;
    
    model->range.min = 0.5;
    model->range.max = 130.0;
    return model;
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: Optix Simulation" << std::endl;

    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " [meshfile]" << std::endl;
        return 0;
    }

    StopWatch sw;
    double el;

    sw();
    OptixMapPtr map = importOptixMap(argv[1]);
    el = sw();
    std::cout << argv[1] << ": loaded in " << el << "s" << std::endl;

    
    SphereSimulatorOptix sim(map);

    // Define and set Scanner Model
    auto model = velodyne_model();

    sim.setModel(model);

    // Define and set Transformation between sensor and base 
    // (sensor offset to poses)
    Memory<Transform, RAM> Tsb(1);
    Tsb->R.x = 0.0;
    Tsb->R.y = 0.0;
    Tsb->R.z = 0.0;
    Tsb->R.w = 1.0;
    Tsb->t.x = 0.01;
    Tsb->t.y = 0.0;
    Tsb->t.z = 0.0; // offset on z axis

    sim.setTsb(Tsb);


    // Define and set poses to transform from
    // Transformations between base and map
    size_t Nposes = 10000;



    size_t Nrays = Nposes * model->theta.size * model->phi.size;

    Memory<Transform, RAM> Tbm(Nposes);
    for(size_t i=0; i<Nposes; i++)
    {
        Tbm[i].R.x = 0.0;
        Tbm[i].R.y = 0.0;
        Tbm[i].R.z = 0.0;
        Tbm[i].R.w = 1.0;
        Tbm[i].t.x = 0.1;
        Tbm[i].t.y = 0.0;
        Tbm[i].t.z = 0.0;
    }

    Memory<Transform, VRAM_CUDA> Tbm_gpu;
    Tbm_gpu = Tbm;

    // predefine memory
    Memory<float, VRAM_CUDA> ranges_gpu(Nrays);
    Memory<Vector, VRAM_CUDA> normals_gpu(Nrays);

    sim.simulateRanges(Tbm_gpu, ranges_gpu);

    using ResultT = Bundle<Ranges<VRAM_CUDA> >;

    ResultT res;
    res.ranges.resize(Nrays);

    sim.preBuildProgram<ResultT>();

    double el1, el2;

    std::vector<double> runtimes;

    for(size_t i=0; i<10; i++)
    {
        // sw();
        // sim.simulateRanges(Tbm_gpu, ranges_gpu);
        // cudaDeviceSynchronize();
        // el1 = sw();
        
        sw();
        sim.simulate<ResultT>(Tbm_gpu, res);
        cudaDeviceSynchronize();
        el2 = sw();

        runtimes.push_back(el2);
    }
    

    Memory<float, RAM> ranges;
    ranges = ranges_gpu;

    // std::cout << "Simulated " << Tbm.size() << " poses / " << ranges.size() << " ranges in " << el << "s" << std::endl;
    std::cout << "Result: " << ranges[0] << std::endl;

    return 0;
}