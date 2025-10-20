// Core rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>

// Embree rmagine includes
#include <rmagine/simulation/SphereSimulatorEmbree.hpp>
#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/embree/embree_shapes.h>



using namespace rmagine;




std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<float> dist(0, 1);

Transform randomTransform()
{
    Transform tf;
    tf.R = {dist(e2), dist(e2), dist(e2), dist(e2)};
    tf.R.normalizeInplace();
    tf.t = {dist(e2), dist(e2), dist(e2)};
    tf.t.normalizeInplace();
}

void fillWithRandomTfs(Memory<Transform, RAM>& tbm_ram)
{
    for(size_t i = 0; i < tbm_ram.size(); i++)
    {
        tbm_ram[i] = randomTransform();
    }
}



void printAsPoints(std::vector<double>& results)
{
    for(size_t i = 0; i < results.size(); i++)
    {
        std::cout << "("<< i+1 << ", " << results[i] << "),";
    }
    std::cout << std::endl;
}



EmbreeMapPtr make_sphere_map(unsigned int num_long, unsigned int num_lat)
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();

    EmbreeMeshPtr mesh = std::make_shared<EmbreeSphere>(num_long, num_lat);
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    std::cout << "Num Faces: " << mesh->faces.size() << std::endl;

    return std::make_shared<EmbreeMap>(scene);
}



size_t reps = 100;

size_t num_maps = 10;
size_t map_param = 100;

size_t num_tbms = 10;
size_t tbm_param = 1000;

int main(int argc, char** argv)
{
    SphericalModel sphereSensor = SphericalModel();
    sphereSensor.theta.min = -M_PI;
    sphereSensor.theta.inc = 0.4 * M_PI / 180.0;
    sphereSensor.theta.size = 900;
    sphereSensor.phi.min = -15.0 * M_PI / 180.0;
    sphereSensor.phi.inc = 2.0 * M_PI / 180.0;
    sphereSensor.phi.size = 16;
    sphereSensor.range.min = 0.1;
    sphereSensor.range.max = 130.0;
    Memory<SphericalModel, RAM> sphereSensor_ram(1);
    sphereSensor_ram[0] = sphereSensor;
    std::cout << "Unit: 1 Velodyne scan (velo) = " << sphereSensor.theta.size*sphereSensor.phi.size << " Rays" << std::endl;


    Transform tsb = Transform();
    tsb.setIdentity();
    Memory<Transform, RAM> tsb_ram(1);
    tsb_ram[0] = tsb;


    SphereSimulatorEmbreePtr sim_gpu_sphere = std::make_shared<SphereSimulatorEmbree>();
    sim_gpu_sphere->setModel(sphereSensor_ram);
    sim_gpu_sphere->setTsb(tsb_ram);


    using ResultT = Bundle<
        Ranges<RAM> 
        // ,Hits<RAM>
        // ,Points<RAM>
        // ,Normals<RAM>
        // ,FaceIds<RAM>
        // ,GeomIds<RAM>
        // ,ObjectIds<RAM>
    >;


    rmagine::StopWatch sw;
    std::cout << "\n" << std::endl;



    //measure different num of sensors
    {
        EmbreeMapPtr map = make_sphere_map(map_param*num_maps, map_param*num_maps);
        sim_gpu_sphere->setMap(map);


        std::vector<double> results;
        for(size_t i = 1; i <= num_tbms; i++)
        {
            Memory<Transform, RAM> tbm_ram(tbm_param*i);
            for(size_t i = 0; i < tbm_ram.size(); i++)
            {
                tbm_ram[i] = tsb;
            }
            std::cout << "Num Poses: " << tbm_ram.size() << std::endl;


            ResultT res2;
            res2.ranges.resize(tbm_ram.size()*sphereSensor.phi.size*sphereSensor.theta.size);


            //prebuild
            sim_gpu_sphere->simulate(tbm_ram, res2);


            //measure
            double elapsed = 0.0;
            double elapsed_total = 0.0;
            std::cout << "-- Starting Measurement --" << std::endl;
            for(size_t j = 1; j <= reps; j++)
            {
                fillWithRandomTfs(tbm_ram);

                sw();
                sim_gpu_sphere->simulate(tbm_ram, res2);
                double elapsed = sw();
                elapsed_total += elapsed;
                
                std::cout 
                << std::fixed
                << "[Elapsed: " << elapsed << "; "
                << "Elapsed Total: " << elapsed_total << "; "
                << "Elapsed Average: " << elapsed_total/(static_cast<double>(j)) << "]" << ((j==reps) ? "" : "\r");
                std::cout.flush();
            }
            results.push_back(elapsed_total/(static_cast<double>(reps)));
            std::cout << "\n" << std::endl;
        }
        printAsPoints(results);
    }


    std::cout << "\n" << std::endl;


    //measure different num of faces
    {
        Memory<Transform, RAM> tbm_ram(tbm_param*num_tbms);
        for(size_t i = 0; i < tbm_ram.size(); i++)
        {
            tbm_ram[i] = tsb;
        }


        ResultT res;
        res.ranges.resize(tbm_ram.size()*sphereSensor.phi.size*sphereSensor.theta.size);


        std::vector<double> results;
        for(size_t i = 1; i <= num_maps; i++)
        {
            EmbreeMapPtr map = make_sphere_map(map_param*i, map_param*i);
            sim_gpu_sphere->setMap(map);


            //prebuild
            sim_gpu_sphere->simulate(tbm_ram, res);


            //measure
            double elapsed = 0.0;
            double elapsed_total = 0.0;
            std::cout << "-- Starting Measurement --" << std::endl;
            for(size_t j = 1; j <= reps; j++)
            {
                fillWithRandomTfs(tbm_ram);

                sw();
                sim_gpu_sphere->simulate(tbm_ram, res);
                elapsed = sw();
                elapsed_total += elapsed;
                
                std::cout 
                << std::fixed
                << "[Elapsed: " << elapsed << "; "
                << "Elapsed Total: " << elapsed_total << "; "
                << "Elapsed Average: " << elapsed_total/(static_cast<double>(j)) << "]" << ((j==reps) ? "" : "\r");
                std::cout.flush();
            }
            results.push_back(elapsed_total/(static_cast<double>(reps)));
            std::cout << "\n" << std::endl;
        }
        std::cout << "" << std::endl;
        printAsPoints(results);
    }

    std::cout << "\nFinished." << std::endl;

    return EXIT_SUCCESS;
}
