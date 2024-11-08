#include <iostream>


#include <rmagine/math/types.h>

#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <rmagine/math/statistics.h>

#include <cassert>

#include <rmagine/math/omp.h>

#include <algorithm>
#include <random>

namespace rm = rmagine;

size_t n_points = 1e9;

template<typename T>
bool is_valid(T a)
{
    return a == a;
}

void printStats(rm::CrossStatistics_<double> stats)
{
    std::cout << "CrossStatistics_<double>: " << std::endl;
    std::cout << "- dataset mean: " << stats.dataset_mean << std::endl;
    std::cout << "- model mean: " << stats.model_mean << std::endl;
    std::cout << "- cov: " << stats.covariance << std::endl;
    std::cout << "- n meas: " << stats.n_meas << std::endl; 
}

void checkStats(rm::CrossStatistics_<double> stats)
{
    // check for nans
    if(!is_valid(stats.dataset_mean.x)){throw std::runtime_error("ERROR: NAN");};
    if(!is_valid(stats.dataset_mean.y)){throw std::runtime_error("ERROR: NAN");};
    if(!is_valid(stats.dataset_mean.z)){throw std::runtime_error("ERROR: NAN");};
    
    if(!is_valid(stats.model_mean.x)){throw std::runtime_error("ERROR: NAN");};
    if(!is_valid(stats.model_mean.y)){throw std::runtime_error("ERROR: NAN");};
    if(!is_valid(stats.model_mean.z)){throw std::runtime_error("ERROR: NAN");};

    if(!is_valid(stats.covariance(0,0))){throw std::runtime_error("ERROR: NAN");};
    if(!is_valid(stats.covariance(0,1))){throw std::runtime_error("ERROR: NAN");};
    if(!is_valid(stats.covariance(0,2))){throw std::runtime_error("ERROR: NAN");};

    if(!is_valid(stats.covariance(1,0))){throw std::runtime_error("ERROR: NAN");};
    if(!is_valid(stats.covariance(1,1))){throw std::runtime_error("ERROR: NAN");};
    if(!is_valid(stats.covariance(1,2))){throw std::runtime_error("ERROR: NAN");};

    if(!is_valid(stats.covariance(2,0))){throw std::runtime_error("ERROR: NAN");};
    if(!is_valid(stats.covariance(2,1))){throw std::runtime_error("ERROR: NAN");};
    if(!is_valid(stats.covariance(2,2))){throw std::runtime_error("ERROR: NAN");};
}

bool equal(rm::CrossStatistics_<double> a, rm::CrossStatistics_<double> b)
{
    if(a.n_meas != b.n_meas)
    {
        return false;
    }

    if( (a.dataset_mean - b.dataset_mean).l2norm() > 0.0001)
    {
        return false;
    }

    if( (a.model_mean - b.model_mean).l2norm() > 0.0001)
    {
        return false;
    }

    rm::Matrix_<double, 3, 3> cov_diff = a.covariance - b.covariance;
    double cov_diff_abs_sum = 0.0;

    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            cov_diff_abs_sum += fabs(cov_diff(i, j));
        }
    }
    
    cov_diff_abs_sum /= 9.0;

    if(cov_diff_abs_sum > 0.0001)
    {
        return false;
    }
    
    return true;
}

void test_incremental()
{
    rm::StopWatch sw;
    double el;
    rm::CrossStatistics_<double> total1 = rm::CrossStatistics_<double>::Identity();

    sw();

    for(size_t i=0; i < n_points; i++)
    {
        double p = static_cast<double>(i) / static_cast<double>(n_points);
        rm::Vector3_<double> d = {-p, p*10.f, p};
        rm::Vector3_<double> m = {p, p, -p};
        const rm::CrossStatistics_<double> local = rm::CrossStatistics_<double>::Init(d, m);
        total1 += rm::CrossStatistics_<double>::Init(d, m);
    }

    el = sw();

    // std::cout << "Incremental: " << el << std::endl; // 0.18
    // printStats(total1);
    assert(total1.n_meas == n_points);

    rm::CrossStatistics_<double> total2 = rm::CrossStatistics_<double>::Identity();
    for(size_t i=0; i < n_points; i++)
    {
        double p = static_cast<double>(i) / static_cast<double>(n_points);
        rm::Vector3_<double> d = {-p, p*10.f, p};
        rm::Vector3_<double> m = {p, p, -p};
        const rm::CrossStatistics_<double> local = rm::CrossStatistics_<double>::Init(d, m);
        total2 = total2 + local;
    }
    

    if(!equal(total1, total2))
    {
        std::cout << "MUST BE THE SAME" << std::endl;
    
        printStats(total1);
        printStats(total2);

        throw std::runtime_error("test_incremental - += and + operator produce different results");
    }
    printStats(total1);

    std::cout << "test_incremental() - success. runtime: " << el << " s" << std::endl;
}


std::vector<rm::CrossStatistics_<double>> init_reduction(
  rm::Memory<rm::Vector3_<double>> dataset,
  rm::Memory<rm::Vector3_<double>> model,
  std::vector<int> ids)
{
    std::vector<rm::CrossStatistics_<double>> ret;

    for(size_t i = 0; i<ids.size(); i++)
    {
        ret.push_back(rm::CrossStatistics_<double>::Init(dataset[ids[i]], model[ids[i]]));
    }

    return ret;
}

std::vector<rm::CrossStatistics_<double>> reduce_once(
  std::vector<rm::CrossStatistics_<double>> data)
{
    std::vector<rm::CrossStatistics_<double>> ret;

    for(size_t i=1; i<data.size(); i+=2)
    {
        ret.push_back(data[i] + data[i-1]);
    }

    if(data.size() % 2)
    {
        ret.push_back(data.back());
    }

    return ret;
}

rm::CrossStatistics_<double> reduce(std::vector<rm::CrossStatistics_<double>> data)
{
    while(data.size() > 1)
    {
        data = reduce_once(data);
    }
    return data[0];
}

void test_reduction_order()
{
    size_t num_elements = n_points;
    size_t num_shuffles = 100;

    rm::Memory<rm::Vector3_<double>> dataset(num_elements);
    rm::Memory<rm::Vector3_<double>> model(num_elements);

    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for(size_t i=0; i<num_elements; i++)
    {
        dataset[i].x = dist(g);
        dataset[i].y = dist(g);
        dataset[i].z = dist(g);

        model[i].x = dist(g);
        model[i].y = dist(g);
        model[i].z = dist(g);
    }

    std::vector<int> ids(num_elements);
    std::iota(ids.begin(), ids.end(), 0);

    std::vector<rm::CrossStatistics_<double>> data = init_reduction(dataset, model, ids);
    rm::CrossStatistics_<double> stats = reduce(data);
    
    for(size_t tries = 0; tries < num_shuffles; tries++)
    {
        std::shuffle(ids.begin(), ids.end(), g);

        std::vector<rm::CrossStatistics_<double>> data_inner = init_reduction(dataset, model, ids);
        rm::CrossStatistics_<double> stats_inner = reduce(data_inner);

        if(!equal(stats, stats_inner))
        {
            std::cout << "The order of the reduction shouldn't matter! Something is wrong here" << std::endl;
            
            std::cout << "Stats Outer: " << std::endl;
            printStats(stats);

            std::cout << "Stats Inner: " << std::endl;
            printStats(stats_inner);
            throw std::runtime_error("REDUCTION SENSITIVE TO ORDER");
        }
    }

    std::cout << "test_reduction() success!" << std::endl;
}


void test_paper()
{
    std::cout << "Recreating Paper Example MICP-L Fig 3." << std::endl;
  
    rm::CrossStatistics_<double> c11, c12, c13, c14;
    
    c11.dataset_mean = {1.0, 0.0, 0.0};
    c11.model_mean   = {3.0, 0.0, 0.0};
    c11.covariance.setZeros();
    c11.n_meas = 1;

    c12.dataset_mean = {5.0, 0.0, 0.0};
    c12.model_mean   = {1.0, 0.0, 0.0};
    c12.covariance.setZeros();
    c12.n_meas = 1;

    c13.dataset_mean = {6.0, 0.0, 0.0};
    c13.model_mean   = {4.0, 0.0, 0.0};
    c13.covariance.setZeros();
    c13.n_meas = 1;

    c14.dataset_mean = {8.0, 0.0, 0.0};
    c14.model_mean   = {8.0, 0.0, 0.0};
    c14.covariance.setZeros();
    c14.n_meas = 1;

    // make the reduction: second level of the tree
    rm::CrossStatistics_<double> c21, c22;

    // the next two lines can be executed in parallel
    c21 = c11 + c12;
    c22 = c13 + c14;

    // third level
    rm::CrossStatistics_<double> c31;
    c31 = c21 + c22;

    printStats(c31);

    rm::CrossStatistics_<double> gt = rm::CrossStatistics_<double>::Identity();
    gt.dataset_mean = {5.0, 0.0, 0.0};
    gt.model_mean = {4.0, 0.0, 0.0};
    gt.covariance.setZeros();
    gt.covariance(0,0) = 4.0;
    gt.n_meas = 4;

    if(!equal(c31, gt))
    {
        std::cout << "ERROR" << std::endl;
        std::cout << "Result:" << std::endl;
        printStats(c31);

        std::cout << "GT:"  << std::endl;
        printStats(gt);
        throw std::runtime_error("PAPER EXAMPLE WONT WORK WITH THIS CODE!");
    }

    std::cout << "test_paper() success!" << std::endl;
}


// from here on everything goes wrong

void test_parallel_reduce()
{
    rm::StopWatch sw;
    double el;
    rm::CrossStatistics_<double> total1 = rm::CrossStatistics_<double>::Identity();

    sw();
    rm::CrossStatistics_<double> c = rm::CrossStatistics_<double>::Identity();

    #pragma omp parallel for reduction(+: total1)
    for(size_t i=0; i<n_points; i++)
    {
        double p = static_cast<double>(i) / static_cast<double>(n_points);
        rm::Vector3_<double> d = {-p, p*10.f, p};
        rm::Vector3_<double> m = {p, p, -p};

        total1 += rm::CrossStatistics_<double>::Init(d, m);

        //p = static_cast<double>(i) / static_cast<double>(n_points);
        //rm::Vector3_<double> d = {1.0, 0.0, 1.0};
        //rm::Vector3_<double> m = {0.0, p, 0.0};
        //rm::CrossStatistics_<double> y = rm::CrossStatistics_<double>::Init(d, m);
        //rm::CrossStatistics_<double> t = total1 + y0;
        //c = (t - total1) - y;
        //total1 = t;
//        total1 += rm::CrossStatistics_<double>::Init(d, m);
    }

    el = sw();

    std::cout << "Parallel Reduce: " << el << std::endl; // 0.02

    printStats(total1);
    assert(total1.n_meas == n_points);

    // compare to sequenctial results
    rm::CrossStatistics_<double> total2 = rm::CrossStatistics_<double>::Identity();
    for(size_t i=0; i<n_points; i++)
    {
        double p = static_cast<double>(i) / static_cast<double>(n_points);
        rm::Vector3_<double> d = {-p, p*10.f, p};
        rm::Vector3_<double> m = {p, p, -p};

        total2 += rm::CrossStatistics_<double>::Init(d, m);
    }

    printStats(total2);

    if(!equal(total1, total2))
    {
        throw std::runtime_error("test_parallel_reduce - parallel reduce gives different results as sequential");
    } // different results?
}


void test_func()
{
    // rm::StopWatch sw;
    // double el;
    
    // rm::CrossStatistics_<double> stats_tmp = rm::CrossStatistics_<double>::Identity();
    // checkStats(stats_tmp);


    // rm::Memory<rm::Vector3_<double>> dataset_points(n_points);
    // rm::Memory<rm::Vector3_<double>> model_points(n_points);
    // rm::Memory<unsigned int> dataset_mask(n_points);
    // rm::Memory<unsigned int> dataset_ids(n_points);

    // // rm::Memory<unsigned int> mask;

    // // fill
    // for(size_t i=0; i<n_points; i++)
    // {
    //     rm::Vector3_<double> d = {1.0, 0.0, 1.0};
    //     rm::Vector3_<double> m = {0.0, 1.0, 0.0};

    //     dataset_points[i] = d;
    //     model_points[i] = m;

    //     dataset_mask[i] = i%2;
    //     dataset_ids[i] = i%4; // 0,1,2,3
    //     // dataset_ids[i] = 0;
    // }

    // dataset_ids[1] = 2;

    // ////
    // // mask: 0 1 0 1 0 1 0 1
    // // ids:  0 1 2 3 0 1 2 3
    // std::cout << "Define dataset" << std::endl;

    // // // define dataset and model from given memory
    // // rm::PointCloudView dataset = {.points = dataset_points};

    // // // std::cout << "Define model" << std::endl;
    // // rm::PointCloudView model = {.points = model_points};

    // rm::Transform Tpre = rm::Transform::Identity();

    // rm::UmeyamaReductionConstraints params;
    // params.max_dist = 2.0;

    // // results
    // rm::CrossStatistics_<double> stats;
    
    // std::cout << "RUN!" << std::endl;

    // sw();
    // stats = rm::statistics_p2p(Tpre, dataset, model, params);
    // el = sw();    

    // std::cout << "statistics_p2p: " << el << " s" << std::endl;

    // printStats(stats);
    // checkStats(stats);
    // if(stats.n_meas != n_points){throw std::runtime_error("ERROR: Too many points");}

       
    // sw();
    // stats = rm::statistics_p2p(Tpre, 
    //     {.points = dataset_points, .mask=dataset_mask},  // dataset
    //     {.points = model_points}, // model
    //     params);
    // el = sw();

    // printStats(stats);
    // checkStats(stats);
    // if(stats.n_meas != n_points/2){throw std::runtime_error("ERROR: Too many points");}

    // params.dataset_id = 2;
    // sw();
    // // stats = rm::statistics_p2p(Tpre, 
    // //     {.points = dataset_points, .mask=dataset_mask, .ids=dataset_ids},  // dataset
    // //     {.points = model_points}, // model
    // //     params);
    // // el = sw();

    // // printStats(stats);
    // // checkStats(stats);
    // if(stats.n_meas != 0){throw std::runtime_error("ERROR: Too many points");}
    
}


int main(int argc, char** argv)
{
    srand((unsigned int) time(0));

    std::cout << "STATS TEST" << std::endl;

    // This is essentially checking if the math is correct
    test_incremental();
    // test_reduction_order();
    // test_paper();


    // THIS goes wrong!
    test_parallel_reduce();
    test_func();

    return 0;
}

