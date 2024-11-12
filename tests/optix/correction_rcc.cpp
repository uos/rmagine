#include <iostream>
#include <memory>
#include <cassert>
#include <sstream>

#include <rmagine/map/OptixMap.hpp>
#include <rmagine/map/optix/optix_shapes.h>
#include <rmagine/map/optix/OptixScene.hpp>

#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/types/sensors.h>

#include <rmagine/math/statistics.cuh>
#include <rmagine/math/linalg.h>

#include <rmagine/util/prints.h>
#include <rmagine/util/exceptions.h>

namespace rm = rmagine;

template<typename DataT>
void printStats(rm::CrossStatistics_<DataT> stats)
{
    std::cout << "CrossStatistics: " << std::endl;
    std::cout << "- dataset mean: " << stats.dataset_mean << std::endl;
    std::cout << "- model mean: " << stats.model_mean << std::endl;
    std::cout << "- cov: " << stats.covariance << std::endl;
    std::cout << "- n meas: " << stats.n_meas << std::endl; 
}

rm::OptixMapPtr make_map()
{
    rm::OptixScenePtr scene = std::make_shared<rm::OptixScene>();

    rm::OptixGeometryPtr mesh = std::make_shared<rm::OptixCube>();
    // mesh->apply();
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    return std::make_shared<rm::OptixMap>(scene);
}

rm::SphericalModel define_sensor_model()
{
    rm::SphericalModel model;
    model.theta.min = -M_PI;
    model.theta.inc = 1.0 * DEG_TO_RAD_F;
    model.theta.size = 360;

    model.phi.min = -64.0 * DEG_TO_RAD_F;
    model.phi.inc = 4.0 * DEG_TO_RAD_F;
    model.phi.size = 32;
    
    model.range.min = 0.5;
    model.range.max = 130.0;
    return model;
}

template<typename Tfrom, typename Tto>
void cast(rm::MemoryView<Tfrom, rm::RAM> from, rm::MemoryView<Tto, rm::RAM> to)
{
    for(size_t i=0; i<from.size(); i++)
    {
        to[i] = static_cast<Tto>(from[i]);
    }
}

unsigned int count(rm::MemoryView<unsigned int, rm::RAM> data)
{
    unsigned int ret = 0;
    for(size_t i=0; i<data.size(); i++)
    {
        ret += data[i];
    }
    return ret;
}

int main(int argc, char** argv)
{
    std::cout << "Correction OptiX-RCC + GPU optimization" << std::endl;

    // create data
    rm::SphereSimulatorOptix sim;

    rm::OptixMapPtr map = make_map();
    sim.setMap(map);
    
    auto sensor_model = define_sensor_model();
    sim.setModel(sensor_model);

    rm::Transform Tbm_gt = rm::Transform::Identity();

    rm::Memory<rm::Transform, rm::RAM> Tbm(1);
    Tbm[0] = Tbm_gt;

    // define what we want to simulate and pre-malloc all buffers
    rm::IntAttrAll<rm::VRAM_CUDA> dataset;
    rm::resize_memory_bundle<rm::VRAM_CUDA>(dataset, sensor_model.getWidth(), sensor_model.getHeight(), 1);
    
    sim.setTsb(rm::Transform::Identity());
    sim.simulate(Tbm, dataset);

    rm::PointCloudView_<rm::VRAM_CUDA> cloud_dataset = {
        .points = dataset.points,
        .mask = dataset.hits
    };

    ////////////////////////////
    // MICP params
    // correspondence searches
    size_t n_outer = 5;
    // optimization steps using the same correspondences
    size_t n_inner = 5;
    rm::UmeyamaReductionConstraints params;
    params.max_dist = 100.0;
    /////////////////////////////
    
    // pose of robot
    rm::Transform Tbm_est = rm::Transform::Identity();
    Tbm_est.t.z = 0.1; // perturbe the pose

    std::cout << "0: " << Tbm_est << " -> " << Tbm_gt << std::endl;

    // pre-create buffers for RCC
    rm::IntAttrAll<rm::VRAM_CUDA> model;
    rm::resize_memory_bundle<rm::VRAM_CUDA>(model, sensor_model.getWidth(), sensor_model.getHeight(), 1);

    for(size_t i=0; i<n_outer; i++)
    {
        // find RCC at estimated pose of robot
        rm::MemoryView<rm::Transform> Tbm_est_view(&Tbm_est, 1);
        sim.simulate(Tbm_est_view, model);

        rm::PointCloudView_<rm::VRAM_CUDA> cloud_model = {
            .points = model.points,
            .mask = model.hits,
            .normals = model.normals
        };

        // this describes a transformation in time of the base frame
        // source: base, t+1 (after registration)
        // target: base, t   (before registration)
        rm::Transform Tpre = rm::Transform::Identity();
        for(size_t j=0; j<n_inner; j++)
        {
            rm::CrossStatistics stats = rm::statistics_p2l(Tpre, cloud_dataset, cloud_model, params);

            // printStats(stats);

            rm::Transform Tpre_next = rm::umeyama_transform(stats);
            Tpre = Tpre * Tpre_next;
        }

        // to get the final transform from (base, t+1) to map we first have to transform 
        // from (base,t+1) -> (base,t) before transforming from (base,t) -> (map)
        Tbm_est = Tbm_est * Tpre;
        std::cout << i+1 << ": " << Tbm_est << " -> " << Tbm_gt << std::endl;
    }

    // diff from one base frame to the other
    // transform from gt to estimation base frame
    auto Tdiff = ~Tbm_est * Tbm_gt;
    if(fabs(Tdiff.t.z) > 0.001)
    {
        std::stringstream ss;
        ss << "Embree Correction RCC results wrong!";
        RM_THROW(rm::EmbreeException, ss.str());
    }

    return 0;
}