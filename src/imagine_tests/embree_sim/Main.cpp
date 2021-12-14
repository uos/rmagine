#include <iostream>
#include <imagine/simulation/SphereSimulatorEmbree.hpp>
#include <imagine/util/StopWatch.hpp>

#include <imagine/types/Bundle.hpp>
#include <imagine/simulation/SimulationResults.hpp>

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


int main(int argc, char** argv)
{
    std::cout << "Imagine Test: Embree Simulator" << std::endl;
    
    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " [meshfile]" << std::endl;
        return 0;
    }

    StopWatch sw;
    double el;

    sw();
    EmbreeMapPtr map = importEmbreeMap(argv[1]);
    el = sw();
    std::cout << argv[1] << ": loaded in " << el << "s" << std::endl;

    SphereSimulatorEmbree sim(map);

    // Define and set Scanner Model
    Memory<LiDARModel, RAM> model = velodyne_model();
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
    size_t Nposes = 1;
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

    sw();
    Memory<float, RAM> ranges = sim.simulateRanges(Tbm);
    el = sw();

    std::cout << "Simulated " << Tbm.size() << " poses / " << ranges.size() << " ranges in " << el << "s" << std::endl;

    std::cout << "Result: " << ranges[0] << std::endl;


    // Generic API
    using SimulatedT = Bundle<Hits<RAM> >;

    SimulatedT res = sim.simulate<SimulatedT>(Tbm);

    std::cout << "first ray hit?: " << (unsigned int)res.hits[0] << std::endl;
    


    return 0;
}