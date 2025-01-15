#include <iostream>
#include <memory>
#include <cassert>
#include <sstream>

#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/map/embree/embree_shapes.h>
#include <rmagine/map/embree/EmbreeScene.hpp>


#include <rmagine/simulation/SphereSimulatorEmbree.hpp>
#include <rmagine/types/sensors.h>

#include <rmagine/math/statistics.h>
#include <rmagine/math/linalg.h>

#include <rmagine/util/prints.h>
#include <rmagine/util/exceptions.h>



namespace rm = rmagine;

// template<typename DataT>
// void printStats(rm::CrossStatistics_<DataT> stats)
// {
//     std::cout << "CrossStatistics: " << std::endl;
//     std::cout << "- dataset mean: " << stats.dataset_mean << std::endl;
//     std::cout << "- model mean: " << stats.model_mean << std::endl;
//     std::cout << "- cov: " << stats.covariance << std::endl;
//     std::cout << "- n meas: " << stats.n_meas << std::endl; 
// }

rm::EmbreeMapPtr make_map()
{
    rm::EmbreeScenePtr scene = std::make_shared<rm::EmbreeScene>();

    rm::EmbreeGeometryPtr mesh = std::make_shared<rm::EmbreeCube>();
    // mesh->apply();
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    return std::make_shared<rm::EmbreeMap>(scene);
}

template<typename Tfrom, typename Tto>
void cast(rm::MemoryView<Tfrom, rm::RAM> from, rm::MemoryView<Tto, rm::RAM> to)
{
    for(size_t i=0; i<from.size(); i++)
    {
        to[i] = static_cast<Tto>(from[i]);
    }
}

unsigned int count(rm::MemoryView<uint8_t, rm::RAM> data)
{
    unsigned int ret = 0;
    for(size_t i=0; i<data.size(); i++)
    {
        ret += data[i];
    }
    return ret;
}

// void printCorrespondences(
//     const rm::PointCloudView& cloud_dataset,
//     const rm::PointCloudView& cloud_model)
// {
//     std::cout << cloud_dataset.points.size() << " to " << cloud_model.points.size() << std::endl;
//     for(size_t i=0; i<cloud_dataset.points.size(); i++)
//     {
//         std::cout << cloud_dataset.points[i] << " -> " << cloud_model.points[i] << std::endl;
//     }
// }

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

int main(int argc, char** argv)
{
    std::cout << "Correction Embree-RCC + CPU optimization" << std::endl;

    // create data
    rm::SphereSimulatorEmbree sim;

    rm::EmbreeMapPtr map = make_map();
    sim.setMap(map);
    
    auto sensor_model = define_sensor_model();
    sim.setModel(sensor_model);

    rm::Transform Tbm_gt = rm::Transform::Identity();

    rm::Memory<rm::Transform, rm::RAM> Tbm(1);
    Tbm[0] = Tbm_gt;

    // define what we want to simulate and pre-malloc all buffers
    rm::IntAttrAll<rm::RAM> dataset;
    rm::resize_memory_bundle<rm::RAM>(dataset, sensor_model.getWidth(), sensor_model.getHeight(), 1);
    
    sim.setTsb(rm::Transform::Identity());
    sim.simulate(Tbm, dataset);
    
    rm::PointCloudView cloud_dataset = {
        .points = dataset.points,
        .mask = dataset.hits
    };

    assert(count(dataset.hits) > 0);


    /////////////////////////
    // MICP params
    // correspondence searches
    size_t n_outer = 10;
    // optimization steps using the same correspondences
    size_t n_inner = 5;
    rm::UmeyamaReductionConstraints params;
    params.max_dist = 100.0;
    ///////////////////////////

    // pose of robot
    rm::Transform Tbm_est = rm::Transform::Identity();
    Tbm_est.t.z = 0.1; // perturbe the pose
    Tbm_est.R = rm::EulerAngles{0.0, 0.0, 0.1};
    
    std::cout << "0: " << Tbm_est << " -> " << Tbm_gt << std::endl;

    // pre-create buffers for RCC
    rm::IntAttrAll<rm::RAM> model;
    rm::resize_memory_bundle<rm::RAM>(model, sensor_model.getWidth(), sensor_model.getHeight(), 1);

    for(size_t i=0; i<n_outer; i++)
    {
        // find RCC at estimated pose of robot
        rm::MemoryView<rm::Transform> Tbm_est_view(&Tbm_est, 1);
        sim.simulate(Tbm_est_view, model);

        rm::PointCloudView cloud_model = {
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
    const rm::EulerAngles Ediff = Tdiff.R;
    if(fabs(Tdiff.t.z) > 0.001 || fabs(Ediff.yaw) > 0.001 )
    {
        std::stringstream ss;
        ss << "Embree Correction RCC results wrong!";
        RM_THROW(rm::EmbreeException, ss.str());
    }

    return 0;
}