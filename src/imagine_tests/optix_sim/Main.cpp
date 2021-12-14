#include <iostream>

#include <imagine/simulation/SphereSimulatorOptix.hpp>
#include <imagine/util/StopWatch.hpp>
#include <fstream>

using namespace imagine;

struct MyMesh
{
    unsigned int id;
};

struct MyInstance
{
    unsigned int id;
};

int main(int argc, char** argv)
{
    std::cout << "Imagine Test: Optix Simulation" << std::endl;

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
    Memory<LiDARModel, RAM> model;
    model->theta.min = -M_PI;
    model->theta.max = M_PI; 
    model->theta.size = 440;
    model->theta.step = (model->theta.max - model->theta.min) / ( static_cast<float>(model->theta.size - 1) );
    
    model->phi.min = -M_PI;
    model->phi.max = M_PI;
    model->phi.size = 16;
    // automate this somehow?
    model->phi.step = (model->phi.max - model->phi.min) / ( static_cast<float>(model->phi.size - 1) );
    
    model->range.min = 1.0;
    model->range.max = 100.0;

    sim.setModel(model);

    // Define and set Transformation between sensor and base 
    // (sensor offset to poses)
    Memory<Transform, RAM> Tsb;
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

    for(size_t i=0; i<1000; i++)
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