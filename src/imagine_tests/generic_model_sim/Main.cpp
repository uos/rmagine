#include <iostream>
#include <fstream>
#include <imagine/simulation/O1DnSimulatorEmbree.hpp>
#include <imagine/util/StopWatch.hpp>

// Generic Interface
#include <imagine/simulation/SimulationResults.hpp>
#include <imagine/types/Bundle.hpp>

#include <type_traits>

using namespace imagine;

O1DnModel<RAM> custom_model()
{
    O1DnModel<RAM> model;
    
    size_t N = 1000;

    model.width = N;
    model.height = 1;

    model.range.min = 0.0;
    model.range.max = 100.0;

    model.orig.x = 0.0;
    model.orig.y = 0.0;
    model.orig.z = 0.0;
    model.rays.resize(N);

    for(size_t i=0; i<N; i++)
    {
        float pos = static_cast<float>(i) / 10.0; 
        Vector dir;
        dir.x = 1.0;
        dir.y = sqrt(pos);
        dir.z = sin(pos);
        model.rays[i] = dir.normalized();
    }

    return model;
}

Memory<O1DnModel<RAM>, RAM> custom_model_memory()
{
    Memory<O1DnModel<RAM>, RAM> mymodel(1);
    mymodel[0] = custom_model();
    return mymodel;
}

int main(int argc, char** argv)
{
    std::cout << "Example CPU Custom Models" << std::endl;

    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " [meshfile] " << std::endl;
    }

    // Load Map
    EmbreeMapPtr map = importEmbreeMap(argv[1]);
    std::cout << "Loaded file '" << argv[1] << "'" << std::endl; 

    std::cout << "- Meshes: " << map->meshes.size() << std::endl;

    

    

    // model.resize(0);

    // return 0;


    // Create Simulator in map
    O1DnSimulatorEmbreePtr sim(new O1DnSimulatorEmbree(map) );

    // Define sensor model
    
    // this works
    // auto model_ = custom_model();
    // sim->setModel(model_);
    // auto model = &model_;

    // this not
    auto model = custom_model_memory();
    sim->setModel(model);



    // Define Sensor to base transform (offset between simulated pose and scanner)
    Memory<Transform, RAM> Tsb(1);
    Tsb->R.x = 0.0;
    Tsb->R.y = 0.0;
    Tsb->R.z = 0.0;
    Tsb->R.w = 1.0;
    Tsb->t.x = 0.0;
    Tsb->t.y = 0.0;
    Tsb->t.z = 0.0;
    sim->setTsb(Tsb);

    size_t N = 10;

    // Define poses to simulate from
    Memory<Transform, RAM> Tbm(N);
    for(size_t i=0; i<N; i++)
    {
        // for simplicity take the identity
        Tbm[i] = Tsb[0];
    }

    std::cout << "Simulate Custom Model" << std::endl;

    // simulate ranges and measure time
    StopWatch sw;
    sw();
    Memory<float, RAM> ranges = sim->simulateRanges(Tbm);
    double el = sw();
    std::cout << "Simulated " << N << " sensors in " << el << "s" << std::endl;

    std::ofstream out("example_cpu_custom_model.xyz", std::ios_base::binary);

    if(out.good())
    {
        for(unsigned int vid=0; vid<model->getHeight(); vid++)
        {
            for(unsigned int hid=0; hid<model->getWidth(); hid++)
            {
                const unsigned int loc_id = model->getBufferId(vid, hid);
                Vector orig = model->getOrigin(vid, hid);
                Vector ray = model->getRay(vid, hid);
                
                // std::cout << "Ray: " << ray.x << " " << ray.y << " " << ray.z << std::endl;
                float range = ranges[loc_id];
                if(range >= model->range.min && range <= model->range.max)
                {
                    Point p = orig + ray * range;
                    // std::cout << "Intersection: " << p.x << " " << p.y << " " << p.z << std::endl;
                    out << p.x << " " << p.y << " " << p.z << "\n";
                }
            }
        }

        out.close();
    }

    sim.reset();
    std::cout << "Done" << std::endl;



    return 0;
}