#include <iostream>

#include <imagine/simulation/OptixSimulator.hpp>
#include <imagine/util/StopWatch.hpp>

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

    
    OptixSimulator sim(map);

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
    Memory<float, VRAM_CUDA> ranges_gpu(Nposes * 440 * 16);
    Memory<Vector, VRAM_CUDA> normals_gpu(Nposes * 440 * 16);


    sw();
    sim.simulateRanges(Tbm_gpu, ranges_gpu);
    el = sw();
    std::cout << "OLD: Simulated " << Tbm.size() << " poses / " << ranges_gpu.size() << " ranges in " << el << "s" << std::endl;


    using ResultT = Bundle<Ranges<VRAM_CUDA> >;

    ResultT res;
    res.ranges.resize(Nposes * 440 * 16);
    // res.normals.resize(Nposes * 440 * 16);

    sim.preBuildProgram<ResultT>();

    sw();
    sim.simulate<ResultT>(Tbm_gpu, res);
    el = sw();

    std::cout << "NEW: Simulated " << Tbm.size() << " poses / " << ranges_gpu.size() << " ranges in " << el << "s" << std::endl;


    Memory<float, RAM> ranges;
    ranges = ranges_gpu;

    // std::cout << "Simulated " << Tbm.size() << " poses / " << ranges.size() << " ranges in " << el << "s" << std::endl;
    std::cout << "Result: " << ranges[0] << std::endl;

    return 0;
}