#include <iostream>
#include <rmagine/math/math.h>
#include <rmagine/math/types.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/synthetic.h>


#include <assimp/Exporter.hpp>

#include <rmagine/simulation/SphereSimulatorEmbree.hpp>


using namespace rmagine;

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

void sphereTest()
{
    aiScene scene = genSphere(50, 50);

    Assimp::Exporter exporter;
    exporter.Export(&scene, "ply", "sphere.ply");
}

void cubeTest()
{
    aiScene scene = genCube();

    Assimp::Exporter exporter;
    exporter.Export(&scene, "ply", "cube.ply");
}

void simulationTest()
{
    aiScene scene = genSphere(50, 50);

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
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: Synthetic data" << std::endl;

    sphereTest();
    cubeTest();
    simulationTest();


    return 0;
}