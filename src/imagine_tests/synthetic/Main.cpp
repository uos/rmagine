#include <iostream>
#include <imagine/math/math.h>
#include <imagine/math/types.h>
#include <imagine/util/StopWatch.hpp>
#include <imagine/util/synthetic.h>


#include <assimp/Exporter.hpp>

#include <imagine/simulation/SphereSimulatorEmbree.hpp>


using namespace imagine;

SphericalModel exampleModel()
{
    SphericalModel model;

    model.theta.min = -M_PI;
    model.theta.max = M_PI; 
    model.theta.size = 440;
    model.theta.computeStep();
    
    model.phi.min = -0.261799;
    model.phi.max = 0.261799;
    model.phi.size = 16;
    model.phi.computeStep();
    // automate this somehow?
    
    model.range.min = 0.5;
    model.range.max = 130.0;

    return model;
}

int main(int argc, char** argv)
{
    std::cout << "Imagine Test: Synthetic data" << std::endl;


    aiScene scene = genSphere(50, 50);


    Assimp::Exporter exporter;
    exporter.Export(&scene, "ply", "synthetic.ply");


    // Do Simulation on synthetic data

    EmbreeMapPtr map(new EmbreeMap(&scene));
    SphereSimulatorEmbree sim(map);

    sim.setModel(exampleModel());

    Transform Tsb;
    Tsb.setIdentity();

    sim.setTsb(Tsb);


    Memory<Transform, RAM> Tbm(10);

    for(size_t i=0; i<Tbm.size(); i++)
    {
        Tbm[i].setIdentity();
    }

    auto ranges = sim.simulateRanges(Tbm);
    std::cout << "Simulated " << ranges.size() << " ranges." << std::endl;



    return 0;
}