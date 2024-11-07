#include <iostream>


#include <rmagine/math/types.h>

#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <rmagine/math/statistics.h>

#include <cassert>

#include <rmagine/math/omp.h>

namespace rm = rmagine;

size_t n_points = 10000000;

void test_incremental()
{
    rm::StopWatch sw;
    double el;
    rm::CrossStatistics total = rm::CrossStatistics::Identity();

    sw();

    for(size_t i=0; i<n_points; i++)
    {
        rm::Vector3 d = {1.0, 0.0, 1.0};
        rm::Vector3 m = {0.0, 1.0, 0.0};
        total += rm::CrossStatistics::Init(d, m);
    }

    el = sw();

    std::cout << "Incremental: " << el << std::endl; // 0.18

    std::cout << "Result: " << std::endl;
    std::cout << "- dataset mean: " << total.dataset_mean << std::endl;
    std::cout << "- model mean: " << total.model_mean << std::endl;
    std::cout << "- cov: " << total.covariance << std::endl;
    std::cout << "- n meas: " << total.n_meas << std::endl; 

    assert(total.n_meas == n_points);
}

void test_parallel_reduce()
{
    rm::StopWatch sw;
    double el;
    rm::CrossStatistics total = rm::CrossStatistics::Identity();

    sw();

    #pragma omp parallel for reduction(+: total)
    for(size_t i=0; i<n_points; i++)
    {
        rm::Vector3 d = {1.0, 0.0, 1.0};
        rm::Vector3 m = {0.0, 1.0, 0.0};
        total += rm::CrossStatistics::Init(d, m);
    }

    el = sw();

    std::cout << "Parallel Reduce: " << el << std::endl; // 0.02

    std::cout << "Result: " << std::endl;
    std::cout << "- dataset mean: " << total.dataset_mean << std::endl;
    std::cout << "- model mean: " << total.model_mean << std::endl;
    std::cout << "- cov: " << total.covariance << std::endl;
    std::cout << "- n meas: " << total.n_meas << std::endl; 

    assert(total.n_meas == n_points);
}

template<typename T>
bool is_valid(T a)
{
    return a == a;
}

void checkStats(rm::CrossStatistics stats)
{
    std::cout << "CrossStatistics: " << std::endl;
    std::cout << "- dataset mean: " << stats.dataset_mean << std::endl;
    std::cout << "- model mean: " << stats.model_mean << std::endl;
    std::cout << "- cov: " << stats.covariance << std::endl;
    std::cout << "- n meas: " << stats.n_meas << std::endl; 

    // check for nans

    if(!is_valid(stats.dataset_mean.x)){throw std::runtime_error("ERR");};
    if(!is_valid(stats.dataset_mean.y)){throw std::runtime_error("ERR");};
    if(!is_valid(stats.dataset_mean.z)){throw std::runtime_error("ERR");};
    if(!is_valid(stats.model_mean.x)){throw std::runtime_error("ERR");};
    if(!is_valid(stats.model_mean.y)){throw std::runtime_error("ERR");};
    if(!is_valid(stats.model_mean.z)){throw std::runtime_error("ERR");};


}

void test_func()
{
    rm::StopWatch sw;
    double el;
    

    rm::Memory<rm::Vector> dataset_points(n_points);
    rm::Memory<rm::Vector> model_points(n_points);
    rm::Memory<unsigned int> dataset_mask(n_points);
    rm::Memory<unsigned int> dataset_ids(n_points);

    // rm::Memory<unsigned int> mask;

    // fill
    for(size_t i=0; i<n_points; i++)
    {
        rm::Vector3 d = {1.0, 0.0, 1.0};
        rm::Vector3 m = {0.0, 1.0, 0.0};

        dataset_points[i] = d;
        model_points[i] = m;

        dataset_mask[i] = i%2;
        dataset_ids[i] = i%4; // 0,1,2,3
        // dataset_ids[i] = 0;
    }

    ////
    // mask: 0 1 0 1 0 1 0 1
    // ids:  0 1 2 3 0 1 2 3
    std::cout << "Define dataset" << std::endl;

    // define dataset and model from given memory
    rm::PointCloudView dataset = {.points = dataset_points};

    // std::cout << "Define model" << std::endl;
    rm::PointCloudView model = {.points = model_points};

    rm::Transform Tpre = rm::Transform::Identity();

    rm::UmeyamaReductionConstraints params;
    params.max_dist = 2.0;
    
    std::cout << "RUN!" << std::endl;

    sw();
    rm::CrossStatistics stats = rm::statistics_p2p(Tpre, dataset, model, params);
    el = sw();    




    std::cout << "statistics_p2p: " << el << " s" << std::endl;

    std::cout << n_points << std::endl;

    checkStats(stats);

   



    
    // sw();
    // stats = rm::statistics_p2p(Tpre, 
    //     {.points = dataset_points, .mask=dataset_mask},  // dataset
    //     {.points = model_points}, // model
    //     params);
    // el = sw();

    // std::cout << "Result: " << std::endl;
    // std::cout << "- dataset mean: " << stats.dataset_mean << std::endl;
    // std::cout << "- model mean: " << stats.model_mean << std::endl;
    // std::cout << "- cov: " << stats.covariance << std::endl;
    // std::cout << "- n meas: " << stats.n_meas << std::endl; 


    // params.dataset_id = 2;
    // sw();
    // stats = rm::statistics_p2p(Tpre, 
    //     {.points = dataset_points, .mask=dataset_mask, .ids=dataset_ids},  // dataset
    //     {.points = model_points}, // model
    //     params);
    // el = sw();

    // std::cout << "Result: " << std::endl;
    // std::cout << "- dataset mean: " << stats.dataset_mean << std::endl;
    // std::cout << "- model mean: " << stats.model_mean << std::endl;
    // std::cout << "- cov: " << stats.covariance << std::endl;
    // std::cout << "- n meas: " << stats.n_meas << std::endl; 

    
    
}

int main(int argc, char** argv)
{
    srand((unsigned int) time(0));

    std::cout << "STATS TEST" << std::endl;


    // test_incremental();

    // test_parallel_reduce();

    test_func();


    



    return 0;
}

