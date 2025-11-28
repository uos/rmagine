#include <random>

// Core rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>

// Optix rmagine includes
#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/map/OptixMap.hpp>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/map/optix/optix_shapes.h>


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
    tf.t = tf.t * dist(e2);
    return tf;
}

void fillWithRandomTfs(Memory<Transform, RAM_CUDA>& tbm_ram)
{
    for(size_t i = 0; i < tbm_ram.size(); i++)
    {
        tbm_ram[i] = randomTransform();
    }
}



OptixMapPtr make_sphere_map(unsigned int num_long, unsigned int num_lat)
{
    OptixScenePtr scene = std::make_shared<OptixScene>();

    OptixMeshPtr mesh = std::make_shared<OptixSphere>(num_long, num_lat);
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    return std::make_shared<OptixMap>(scene);
}



size_t reps = 100;

size_t num_maps = 20;
size_t map_param = 500;

size_t num_tbms = 20;
size_t tbm_param = 500;

using ResultT = Bundle<
    Ranges<VRAM_CUDA> 
    // ,Hits<VRAM_CUDA>
    // ,Points<VRAM_CUDA>
    // ,Normals<VRAM_CUDA>
    // ,FaceIds<VRAM_CUDA>
    // ,GeomIds<VRAM_CUDA>
    // ,ObjectIds<VRAM_CUDA>
>;

rmagine::StopWatch sw;


double measure(Memory<Transform, RAM_CUDA>& tbm_ram, Memory<Transform, VRAM_CUDA>& tbm, ResultT& res, OptixMapPtr map, SphereSimulatorOptixPtr sim_gpu_sphere)
{
    //prebuild
    sim_gpu_sphere->simulate(tbm, res);


    std::cout << "Num Poses: " << tbm.size() << std::endl;
    std::cout << "Num Faces: " << map->scene()->geometries().begin()->second->this_shared<OptixMesh>()->faces.size() << std::endl;


    //measure
    double elapsed = 0.0;
    double elapsed_total = 0.0;
    std::cout << "-- Starting Measurement --" << std::endl;
    for(size_t j = 1; j <= reps; j++)
    {
        fillWithRandomTfs(tbm_ram);
        tbm = tbm_ram;

        sw();
        sim_gpu_sphere->simulate(tbm, res);
        double elapsed = sw();
        elapsed_total += elapsed;
        
        std::cout 
        << std::fixed
        << "[Elapsed: " << elapsed << "; "
        << "Elapsed Total: " << elapsed_total << "; "
        << "Elapsed Average: " << elapsed_total/(static_cast<double>(j)) << "]" << ((j==reps) ? "" : "\r");
        std::cout.flush();
    }
    std::stringstream point;


    return elapsed_total/(static_cast<double>(reps));
}


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


    SphereSimulatorOptixPtr sim_gpu_sphere = std::make_shared<SphereSimulatorOptix>();
    sim_gpu_sphere->setModel(sphereSensor_ram);
    sim_gpu_sphere->setTsb(tsb_ram);


    std::cout << "\n" << std::endl;



    //measure different num of sensors
    {
        unsigned int num_lon_and_lat = map_param * sqrt(2.0);
        OptixMapPtr map = make_sphere_map(num_lon_and_lat, num_lon_and_lat);
        sim_gpu_sphere->setMap(map);
        

        std::vector<std::string> results;
        for(size_t i = 0; i <= num_tbms; i++)
        {
            unsigned int tbm_size = tbm_param*i;
            if(i == 0)
                tbm_size = 1;
            Memory<Transform, RAM_CUDA> tbm_ram(tbm_size);
            for(size_t j = 0; j < tbm_ram.size(); j++)
            {
                tbm_ram[j] = tsb;
            }
            Memory<Transform, VRAM_CUDA> tbm(tbm_ram.size());
            tbm = tbm_ram;

            ResultT res;
            res.ranges.resize(tbm.size()*sphereSensor.phi.size*sphereSensor.theta.size);


            double elapsed_avg = measure(tbm_ram, tbm, res, map, sim_gpu_sphere);
            std::stringstream point;
            point << "(" << static_cast<double>(tbm_size)/1000.0 << ", " << elapsed_avg << ")";
            results.push_back(point.str());
            std::cout << "\n" << std::endl;
        }
        for(size_t i = 0; i < results.size(); i++)
        {
            std::cout << results[i];
        }
        std::cout << std::endl;
    }


    std::cout << "\n" << std::endl;


    //measure different num of faces
    {
        Memory<Transform, RAM_CUDA> tbm_ram(tbm_param*num_tbms);
        for(size_t i = 0; i < tbm_ram.size(); i++)
        {
            tbm_ram[i] = tsb;
        }
        Memory<Transform, VRAM_CUDA> tbm(tbm_ram.size());
        tbm = tbm_ram;

        ResultT res;
        res.ranges.resize(tbm.size()*sphereSensor.phi.size*sphereSensor.theta.size);


        std::vector<std::string> results;
        for(size_t i = 0; i <= num_maps; i++)
        {
            unsigned int num_lon_and_lat = static_cast<unsigned int>(static_cast<double>(map_param)*sqrt(static_cast<double>(i)));
            if(i == 0)
                num_lon_and_lat = 10;
            OptixMapPtr map = make_sphere_map(num_lon_and_lat, num_lon_and_lat);
            sim_gpu_sphere->setMap(map);


            double elapsed_avg = measure(tbm_ram, tbm, res, map, sim_gpu_sphere);
            std::stringstream point;
            point << "(" << static_cast<double>(map->scene()->geometries().begin()->second->this_shared<OptixMesh>()->faces.size())/1000000 << ", " << elapsed_avg << ")";
            results.push_back(point.str());
            std::cout << "\n" << std::endl;
        }
        for(size_t i = 0; i < results.size(); i++)
        {
            std::cout << results[i];
        }
        std::cout << std::endl;
    }

    std::cout << "\n\nFinished." << std::endl;

    return EXIT_SUCCESS;
}
